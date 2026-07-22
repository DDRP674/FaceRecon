from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from facereco.celeba import write_split_manifest
from facereco.config import Paths
from facereco.embedding import compute_multinoise_embeddings, export_level_embeddings


def make_paths(args) -> Paths:
    return Paths(
        repo_root=Path(args.repo_root).resolve(),
        data_dir=Path(args.data_dir).resolve(),
        work_dir=Path(args.work_dir).resolve(),
        model_dir=Path(args.model_dir).resolve(),
    )


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", default=Path(__file__).resolve().parents[1] / "data" / "celeba")
    parser.add_argument("--work-dir", default=Path(__file__).resolve().parents[1] / "work")
    parser.add_argument("--model-dir", default=Path(__file__).resolve().parents[1] / "models")


def main() -> None:
    parser = argparse.ArgumentParser(description="CelebA identity defense/attack pipeline")
    add_common(parser)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split")
    p.add_argument("--require-identity", action="store_true")
    p.add_argument("--out", default=None)

    p = sub.add_parser("stage0")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--ctx-id", type=int, default=0)
    p.add_argument("--det-size", type=int, default=320)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-every", type=int, default=8)
    p.add_argument("--embedding-mode", choices=["landmarks", "fast", "full"], default="landmarks")
    p.add_argument("--recognition-batch-size", type=int, default=2048)
    p.add_argument("--preprocess-workers", type=int, default=1)
    p.add_argument("--num-data-shards", type=int, default=1)
    p.add_argument("--data-shard-index", type=int, default=0)
    p.add_argument("--skip-embeddings", action="store_true")
    p.add_argument("--cache-vae", action="store_true")
    p.add_argument("--vae-batch-size", type=int, default=16)
    p.add_argument("--vae-image-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("export-level")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--out", default=None)

    p = sub.add_parser("stage1")
    p.add_argument("--query-split", default="test")
    p.add_argument("--gallery-split", default="train")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--embedding-file", default=None)
    p.add_argument("--out", default=None)

    p = sub.add_parser("stage2")
    p.add_argument("--embedding-file", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--prompt", default=None)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--generated-dir", default=None)

    p = sub.add_parser("stage3")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)

    p = sub.add_parser("export-defended")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")

    p = sub.add_parser("stage4")
    p.add_argument("--embedding-file", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--prompt", default=None)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--generated-dir", default=None)

    p = sub.add_parser("stage5")
    p.add_argument("--embedding-file", required=True)
    p.add_argument("--val-embedding-file", default=None)
    p.add_argument("--eval-embedding-file", default=None)
    p.add_argument("--lora-dir", default=None)
    p.add_argument("--generated-dir", default=None)
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--prompt", default=None)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--evaluate", action="store_true")

    p = sub.add_parser("compare")
    p.add_argument("--stage2-dir", required=True)
    p.add_argument("--stage4-dir", required=True)
    p.add_argument("--stage5-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=8)

    args = parser.parse_args()
    paths = make_paths(args)

    if args.cmd == "split":
        out = Path(args.out) if args.out else paths.work_dir / "celeba_official_split.csv"
        write_split_manifest(paths, out, require_identity=args.require_identity)
        print(out)
    elif args.cmd == "stage0":
        from facereco.celeba import load_records
        from facereco.vae_cache import cache_sdxl_vae_latents

        records = load_records(paths, require_identity=False)
        if args.limit is not None:
            records = records[: args.limit]
        shard_index_offset = 0
        if args.num_data_shards < 1:
            raise ValueError("--num-data-shards must be >= 1")
        if not (0 <= args.data_shard_index < args.num_data_shards):
            raise ValueError("--data-shard-index must be in [0, num_data_shards)")
        if args.num_data_shards > 1:
            total_batches = math.ceil(len(records) / args.batch_size)
            batch_start = total_batches * args.data_shard_index // args.num_data_shards
            batch_end = total_batches * (args.data_shard_index + 1) // args.num_data_shards
            records = records[batch_start * args.batch_size : min(len(records), batch_end * args.batch_size)]
            shard_index_offset = batch_start
        if not args.skip_embeddings:
            compute_multinoise_embeddings(
                records,
                paths,
                batch_size=args.batch_size,
                ctx_id=args.ctx_id,
                det_size=args.det_size,
                overwrite=args.overwrite,
                save_every=args.save_every,
                embedding_mode=args.embedding_mode,
                recognition_batch_size=args.recognition_batch_size,
                preprocess_workers=args.preprocess_workers,
                shard_index_offset=shard_index_offset,
            )
        if args.cache_vae:
            cache_sdxl_vae_latents(
                records,
                paths,
                image_size=args.vae_image_size,
                batch_size=args.vae_batch_size,
                device=args.device,
                overwrite=args.overwrite,
            )
    elif args.cmd == "export-level":
        print(export_level_embeddings(paths, level=args.level, out_path=Path(args.out) if args.out else None))
    elif args.cmd == "stage1":
        from facereco.retrieval import evaluate_retrieval, evaluate_retrieval_file

        out = Path(args.out) if args.out else paths.metrics_dir / "stage1_retrieval.json"
        if args.embedding_file:
            metrics = evaluate_retrieval_file(
                paths,
                Path(args.embedding_file),
                args.query_split,
                args.gallery_split,
                out_json=out,
            )
        else:
            metrics = evaluate_retrieval(paths, args.query_split, args.gallery_split, args.level, out_json=out)
        print(json.dumps(metrics, indent=2))
    elif args.cmd in {"stage2", "stage4"}:
        from facereco.attack import evaluate_reconstructions, generate_from_embedding_file

        emb = Path(args.embedding_file) if args.embedding_file else export_level_embeddings(paths, level=0)
        gen_dir = Path(args.generated_dir) if args.generated_dir else paths.generated_dir / args.cmd
        if args.generate:
            from facereco.attack import DEFAULT_NEGATIVE, DEFAULT_PROMPT

            gen_dir = generate_from_embedding_file(
                paths,
                emb,
                out_dir=gen_dir,
                limit=args.limit,
                width=args.width,
                height=args.height,
                prompt=args.prompt or DEFAULT_PROMPT,
                negative_prompt=args.negative_prompt or DEFAULT_NEGATIVE,
            )
        if args.evaluate:
            out = paths.metrics_dir / f"{args.cmd}_reconstruction.json"
            print(json.dumps(evaluate_reconstructions(paths, gen_dir, limit=args.limit, out_json=out), indent=2))
    elif args.cmd == "stage3":
        from facereco.defense_train import train_defense

        print(train_defense(paths, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr))
    elif args.cmd == "export-defended":
        from facereco.defense_train import export_defended_embeddings

        print(export_defended_embeddings(paths, Path(args.ckpt), split=args.split))
    elif args.cmd == "stage5":
        from facereco.lora_train import train_ip_adapter_lora

        lora_dir = Path(args.lora_dir) if args.lora_dir else paths.checkpoints_dir / "ip_adapter_defended_lora"
        if not args.skip_train:
            lora_dir = train_ip_adapter_lora(
                paths,
                Path(args.embedding_file),
                val_embedding_file=Path(args.val_embedding_file) if args.val_embedding_file else None,
                out_dir=lora_dir,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch or args.steps,
                val_batches=args.val_batches,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            print(lora_dir)
        if args.generate or args.evaluate:
            from facereco.attack import evaluate_reconstructions, generate_from_embedding_file

            if args.eval_embedding_file is None:
                raise ValueError("--eval-embedding-file is required for Stage 5 generation/evaluation.")
            gen_dir = Path(args.generated_dir) if args.generated_dir else paths.generated_dir / "stage5"
            if args.generate:
                from facereco.attack import DEFAULT_NEGATIVE, DEFAULT_PROMPT

                gen_dir = generate_from_embedding_file(
                    paths,
                    Path(args.eval_embedding_file),
                    out_dir=gen_dir,
                    limit=args.limit,
                    width=args.width,
                    height=args.height,
                    prompt=args.prompt or DEFAULT_PROMPT,
                    negative_prompt=args.negative_prompt or DEFAULT_NEGATIVE,
                    lora_dir=lora_dir,
                )
            if args.evaluate:
                out = Path(args.metrics_out) if args.metrics_out else paths.metrics_dir / "stage5_reconstruction.json"
                print(json.dumps(evaluate_reconstructions(paths, gen_dir, limit=args.limit, out_json=out), indent=2))
    elif args.cmd == "compare":
        from facereco.compare import make_reconstruction_comparison

        print(
            make_reconstruction_comparison(
                paths,
                Path(args.stage2_dir),
                Path(args.stage4_dir),
                Path(args.stage5_dir),
                Path(args.out),
                limit=args.limit,
            )
        )


if __name__ == "__main__":
    main()
