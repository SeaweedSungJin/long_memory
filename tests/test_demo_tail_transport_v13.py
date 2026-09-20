"""Actual local ZeroMQ/msgpack transport with a tiny synthetic V13 endpoint."""
from contextlib import redirect_stdout
import io
import queue
import threading
import unittest

from gr00t.policy.server_client import PolicyClient, PolicyServer
from run_scripts.robomme import rollout_demo_tail_v13 as rollout
from tests.test_long_memory_rollout import FakeEnv
from tests.test_rollout_demo_tail_v13 import TailPolicy, config


class TransportPolicy(TailPolicy):
    def get_action(self, observation, options=None):
        return super().get_action(observation, options)

    def ingest_demo_tail(self, **data):
        return self.call_endpoint(rollout.ENDPOINT, data)


class DemoTailTransportTests(unittest.TestCase):
    def run_transport(self, bad=None):
        actor, ready, errors = TransportPolicy(bad), queue.Queue(), []

        def serve():
            server = None
            try:
                # Create/use/close REP socket on its owner thread.
                server = PolicyServer(actor, host="127.0.0.1", port=0)
                server.register_endpoint(rollout.ENDPOINT, actor.ingest_demo_tail)
                import zmq
                port = int(server.socket.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])
                ready.put(port)
                server.run()
            except BaseException as error:
                errors.append(error)
                ready.put(error)
            finally:
                if server is not None:
                    server.socket.close(linger=0)
                    server.context.term()

        worker = threading.Thread(target=serve, daemon=True)
        client = None
        env, records, result = FakeEnv(n_demo=48, finish=20), [], None
        with redirect_stdout(io.StringIO()):
            worker.start()
            try:
                port = ready.get(timeout=5)
                if isinstance(port, BaseException):
                    raise port
                client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=2000)
                import zmq
                client.socket.setsockopt(zmq.RCVTIMEO, 2000)
                client.socket.setsockopt(zmq.SNDTIMEO, 2000)
                client.socket.setsockopt(zmq.LINGER, 0)
                if bad is None:
                    result = rollout.run_episode(client, env, config(), 0, 42, records.append)
                else:
                    with self.assertRaises(RuntimeError):
                        rollout.run_episode(client, env, config(), 0, 42, records.append)
            finally:
                if client is not None:
                    try:
                        client.kill_server()
                    finally:
                        client.socket.close(linger=0)
                        client.context.term()
                worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), "Synthetic transport worker leaked")
        self.assertEqual(errors, [])
        self.assertTrue(env.closed)
        self.assertEqual(len(actor.resets), 2)
        return actor, env, records, result

    def test_uint8_two_camera_payload_roundtrip_and_original_action_loop(self):
        actor, env, records, result = self.run_transport()
        self.assertEqual(result[0]["success"], 1)
        self.assertEqual(len(env.actions), 20)
        self.assertEqual(len(actor.rpc), 1)
        _, payload, calls, noise = actor.rpc[0]
        self.assertEqual((calls, noise), (3, 0))
        self.assertEqual(payload["images"]["front_view"].dtype.name, "uint8")
        self.assertEqual(payload["images"]["front_view"].shape, (15, 8, 8, 3))
        self.assertEqual(len([r for r in records if r["kind"] == "policy_call"]), 5)

    def test_malformed_rpc_response_is_not_successful_bypass(self):
        actor, env, records, _ = self.run_transport("bool")
        self.assertEqual(env.actions, [])
        self.assertEqual(actor.action_noise_draws, 0)
        self.assertTrue(any(r["kind"] == "episode_error" for r in records))
        self.assertFalse(any(r["kind"] == "episode_complete" for r in records))

    def test_real_server_error_response_propagates_and_cleanup_works(self):
        actor, env, records, _ = self.run_transport("exception")
        self.assertEqual(env.actions, [])
        self.assertEqual(actor.action_noise_draws, 0)
        self.assertTrue(any(r["kind"] == "episode_error" for r in records))


if __name__ == "__main__":
    unittest.main()
