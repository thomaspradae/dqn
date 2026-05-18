#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_rewards(path):
    episodes = []
    rewards = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
    if not episodes:
        raise ValueError(f"no rewards found in {path}")
    return episodes, rewards


def rolling_mean(values, window):
    out = []
    running = 0.0
    queue = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        out.append(running / len(queue))
    return out


def nice_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_polyline(draw, points, color, width=1):
    if len(points) < 2:
        return
    draw.line(points, fill=color, width=width, joint="curve")


def main():
    parser = argparse.ArgumentParser(description="Plot DQN training rewards.")
    parser.add_argument("csv", type=Path, help="Path to rewards.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to learning_curve.png next to the CSV.",
    )
    parser.add_argument("--rolling-window", type=int, default=25)
    parser.add_argument("--title", default="DQN CartPole-v1 Learning Curve")
    args = parser.parse_args()

    episodes, rewards = read_rewards(args.csv)
    smoothed = rolling_mean(rewards, args.rolling_window)

    out = args.out or args.csv.with_name("learning_curve.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1400, 850
    margin_left, margin_right = 95, 45
    margin_top, margin_bottom = 95, 95
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    max_reward = max(max(rewards), 500.0)
    y_min, y_max = 0.0, max_reward
    x_min, x_max = min(episodes), max(episodes)
    x_span = max(1, x_max - x_min)
    y_span = max(1.0, y_max - y_min)

    def x_to_px(x):
        return plot_left + int((x - x_min) / x_span * plot_width)

    def y_to_px(y):
        return plot_bottom - int((y - y_min) / y_span * plot_height)

    image = Image.new("RGB", (width, height), "#fbfbf8")
    draw = ImageDraw.Draw(image)
    title_font = nice_font(34)
    label_font = nice_font(22)
    tick_font = nice_font(18)
    small_font = nice_font(17)

    # Background and grid.
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], fill="#ffffff", outline="#1f2933", width=2)
    for i in range(0, 6):
        y_value = y_min + i * y_span / 5
        y = y_to_px(y_value)
        draw.line([plot_left, y, plot_right, y], fill="#e2e8f0", width=1)
        draw.text((22, y - 12), f"{y_value:.0f}", fill="#334155", font=tick_font)
    for i in range(0, 6):
        x_value = x_min + i * x_span / 5
        x = x_to_px(x_value)
        draw.line([x, plot_top, x, plot_bottom], fill="#eef2f7", width=1)
        draw.text((x - 20, plot_bottom + 14), f"{x_value:.0f}", fill="#334155", font=tick_font)

    raw_points = [(x_to_px(x), y_to_px(y)) for x, y in zip(episodes, rewards)]
    smooth_points = [(x_to_px(x), y_to_px(y)) for x, y in zip(episodes, smoothed)]
    draw_polyline(draw, raw_points, "#93c5fd", width=2)
    draw_polyline(draw, smooth_points, "#dc2626", width=4)

    solved_y = y_to_px(500.0)
    draw.line([plot_left, solved_y, plot_right, solved_y], fill="#059669", width=2)
    draw.text((plot_right - 225, solved_y - 28), "CartPole max reward = 500", fill="#047857", font=small_font)

    draw.text((plot_left, 28), args.title, fill="#111827", font=title_font)
    draw.text((plot_left, height - 45), "Episode", fill="#111827", font=label_font)
    draw.text((18, 36), "Reward", fill="#111827", font=label_font)

    legend_x = plot_left + 22
    legend_y = plot_top + 18
    draw.line([legend_x, legend_y, legend_x + 55, legend_y], fill="#93c5fd", width=4)
    draw.text((legend_x + 68, legend_y - 12), "episode reward", fill="#334155", font=small_font)
    draw.line([legend_x, legend_y + 32, legend_x + 55, legend_y + 32], fill="#dc2626", width=5)
    draw.text(
        (legend_x + 68, legend_y + 20),
        f"{args.rolling_window}-episode rolling mean",
        fill="#334155",
        font=small_font,
    )

    last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
    stats = [
        f"episodes: {len(rewards)}",
        f"best: {max(rewards):.0f}",
        f"last-100 avg: {sum(last_100) / len(last_100):.1f}",
        f"final: {rewards[-1]:.0f}",
    ]
    for idx, text in enumerate(stats):
        draw.text((plot_right - 210, plot_top + 18 + idx * 25), text, fill="#334155", font=small_font)

    image.save(out)
    print(out)


if __name__ == "__main__":
    main()
