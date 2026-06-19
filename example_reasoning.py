"""Example of OpenMythos reasoning inference."""
import torch
from open_mythos.main import OpenMythos
from open_mythos.variants import mythos_1b


def main():
    print("=== OpenMythos: Recurrent-Depth Transformer Demonstration ===")

    # ১. MythosConfig সেটআপ এবং ACT, MoE, LoRA কনফিগারেশন
    # আমরা mythos_1b ব্যবহার করে কিছু কাস্টমাইজেশন যোগ করছি
    cfg = mythos_1b()

    # Adaptive Computation Time (ACT) থ্রেশহোল্ড সেট করা
    # এটি নির্ধারণ করে মডেল কখন লুপ থামাবে (যেমন 0.99)
    cfg.act_threshold = 0.99

    # লুপ ইটারেশন সংখ্যা নির্ধারণ (Training time)
    cfg.max_loop_iters = 16

    print("\nConfigured with:")
    print(f"- max_loop_iters: {cfg.max_loop_iters}")
    print(f"- act_threshold: {cfg.act_threshold}")
    print(f"- MoE Experts: {cfg.n_experts} (routed), {cfg.n_shared_experts} (shared)")
    print(f"- LoRA rank: {cfg.lora_rank}")

    # মডেল তৈরি
    model = OpenMythos(cfg)
    model.eval()  # ইনফারেন্স মোডে সেট করা
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ডেমো ইনপুট টোকেনস (Batch size 1, Sequence length 10)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 10))
    print(f"Input shape: {input_ids.shape}")

    # ২. Depth Extrapolation: কঠিন কাজের জন্য লুপ বাড়ানো
    # সাধারণ কাজের জন্য ডিফল্ট লুপ
    print("\n--- Standard Inference ---")
    standard_loops = cfg.max_loop_iters
    out_standard = model.generate(input_ids, max_new_tokens=10, n_loops=standard_loops)
    print(f"Standard generation (loops={standard_loops}) shape: {out_standard.shape}")

    # কঠিন কাজের জন্য বেশি লুপ (Deeper Reasoning)
    print("\n--- Deep Reasoning Inference ---")
    deep_loops = 32  # ট্রেইনিং এর চেয়ে বেশি লুপ
    out_deep = model.generate(input_ids, max_new_tokens=10, n_loops=deep_loops)
    print(f"Deep generation (loops={deep_loops}) shape: {out_deep.shape}")

    # ৩. LTI Injection Verification
    # আমরা পরীক্ষা করে দেখতে পারি যে Injection Matrix A এর Spectral Radius < 1 কি না,
    # যা মডেলের স্ট্যাবিলিটি নিশ্চিত করে।
    A = model.recurrent.injection.get_A()
    rho = torch.linalg.eigvals(A).abs().max().item()
    print(f"\nSpectral radius ρ(A) = {rho:.4f} (Must be < 1 for stability)")
    print(
        "\nDemonstration complete. The model successfully executed standard and extended reasoning loops."
    )


if __name__ == "__main__":
    main()
