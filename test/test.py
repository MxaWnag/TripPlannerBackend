import requests, time, json, statistics, argparse

def bench_once(model, prompt, host="http://localhost:11434", num_predict=512):
    url = f"{host}/api/generate"
    t0 = time.perf_counter()
    with requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": num_predict}
    }, stream=True, timeout=600) as r:
        r.raise_for_status()
        ttft = None
        last = None
        for line in r.iter_lines():
            if not line:
                continue
            obj = json.loads(line.decode("utf-8"))
            # 首个包含内容的分片时间 = TTFT
            if ttft is None and obj.get("response"):
                ttft = time.perf_counter() - t0
            if obj.get("done"):
                last = obj
                break
        t1 = time.perf_counter()

    eval_count = last.get("eval_count", 0)
    eval_duration = last.get("eval_duration", 0)  # ns
    prompt_eval_count = last.get("prompt_eval_count", 0)
    prompt_eval_duration = last.get("prompt_eval_duration", 0)  # ns
    total_duration = last.get("total_duration", int((t1-t0)*1e9))  # 兜底

    tok_s = eval_count / (eval_duration / 1e9) if eval_duration else None
    return {
        "ttft_ms": round((ttft or 0) * 1000, 1),
        "elapsed_ms": round((t1 - t0) * 1000, 1),
        "tok_s": round(tok_s, 1) if tok_s else None,
        "gen_tokens": int(eval_count),
        "prompt_tokens": int(prompt_eval_count),
        "prompt_ms": round(prompt_eval_duration / 1e6, 1),
        "gen_ms": round(eval_duration / 1e6, 1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b-instruct-q4_K_M")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--prompt", default="請用中文自我介紹，控制在200字左右。")
    args = ap.parse_args()

    # 预热
    bench_once(args.model, "warm up", args.host, 16)

    rows = [bench_once(args.model, args.prompt, args.host, args.num_predict) for _ in range(args.repeat)]
    def s(k): return [r[k] for r in rows if r[k] is not None]

    print("Per-run:", rows)
    for k in ["ttft_ms", "elapsed_ms", "tok_s", "gen_tokens", "prompt_tokens"]:
        vals = s(k)
        if not vals: continue
        print(f"{k}: avg={sum(vals)/len(vals):.1f}  p50={statistics.median(vals):.1f}  "
              f"min={min(vals):.1f}  max={max(vals):.1f}")

if __name__ == "__main__":
    main()
