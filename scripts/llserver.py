"""Start / stop a llama.cpp llama-server and send raw-prompt completion requests to it."""
import json, os, subprocess, time, urllib.request

LLAMA_DIR = os.environ.get("LLAMA_DIR", "/workspace/dev-reels/llamacpp/llama-b11870")


class Server:
    def __init__(self, model, n_parallel, ctx_per_slot, port=8091, threads=8, log=None):
        self.port, self.url = port, f"http://127.0.0.1:{port}"
        cmd = [f"{LLAMA_DIR}/llama-server", "-m", model, "-np", str(n_parallel), "-c", str(n_parallel * ctx_per_slot),
               "-t", str(threads), "--port", str(port), "--host", "127.0.0.1", "--no-webui", "-b", "2048",
               "-ub", "512", "--cache-reuse", "0", "--metrics",
               "--cache-ram", "0", "--ctx-checkpoints", "0"]  # keep RAM bounded (an 8 GB prompt cache got the server OOM-killed)
        self.logf = open(log or os.devnull, "a")
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT)
        for _ in range(600):
            try:
                if json.loads(urllib.request.urlopen(self.url + "/health", timeout=2).read()).get("status") == "ok":
                    return
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited; see log " + str(log))
            time.sleep(0.5)
        raise RuntimeError("llama-server did not become healthy")

    def complete(self, prompt, cap, sampling, seed, timeout=7200):
        body = {"prompt": prompt, "n_predict": cap, "seed": seed, "cache_prompt": False, "stream": False,
                "temperature": sampling["temperature"], "top_p": sampling["top_p"], "top_k": sampling["top_k"],
                "min_p": sampling.get("min_p", 0.0), "presence_penalty": sampling.get("presence_penalty", 0.0),
                "frequency_penalty": 0.0, "repeat_penalty": 1.0, "repeat_last_n": cap + 1024, "typical_p": 1.0,
                "dry_multiplier": 0.0, "xtc_probability": 0.0, "n_probs": 0}
        req = urllib.request.Request(self.url + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.logf.close()
