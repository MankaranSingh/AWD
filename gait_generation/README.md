# Gait generation

## Generate one gait

```bash
python3 gait_generator.py -n <name> <--mini> --dx X --dy Y --dt T --length L -o <output_dir>
```

## Generate multiple gaits

```bash
python3 auto_generator.py --bdx_type [go_bdx, mini_bdx, mini2_bdx] --num 100
```

## Replay a move

```bash
python3 replay_amp.py -f <path/.json>
```
