"""Compare fixed-input steering candidates on the Telluride packet's numeric segment extracts."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def latest(rows, times):
  index = np.searchsorted(rows[:, 0], times, side="right") - 1
  selected = rows[index].copy()
  selected[(index < 0) | (times - selected[:, 0] > .05)] = np.nan
  return selected


def intervals(mask):
  edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
  return zip(edges[::2], edges[1::2])


def attenuate(target, limit):
  magnitude = abs(target)
  reduction = .25 * min(magnitude, max(0., (limit - magnitude) / 2))
  return round(target - np.sign(target) * reduction)


def limit(target, last, release, driver):
  low = min(0, max(-384, -384 + 2 * (driver - 50)))
  high = max(0, min(384, 384 + 2 * (driver + 50)))
  target = min(high, max(low, target))
  if last > 0:
    low, high = max(-3, last - release), last + 3
  else:
    low, high = last - 3, min(3, last + release)
  return round(min(high, max(low, target)))


def score_torque(data, totals):
  wire = data["wire"]
  times = wire[:, 0]
  cmd = latest(data["command"], times)
  state = latest(data["state"], times)
  valid = (wire[:, 2] == 1) & (cmd[:, 1] == 1) & (state[:, 2] == 1) & (state[:, 3] == 0)
  for start, end in intervals(valid):
    wanted = np.rint(384 * cmd[start:end, 2])
    recorded = wire[start:end, 1]
    for variant, counts in totals.items():
      last = recorded[0]
      result = []
      for target, driver in zip(wanted, state[start:end, 6]):
        release = 4 if variant == "release4" and abs(driver) <= 150 else 7
        shaped = target
        if variant in ("attenuated", "narrow") and abs(driver) <= 150:
          shaped = attenuate(target, 96 if variant == "attenuated" else 32)
        last = limit(shaped, last, release, driver)
        result.append(last)
      output = np.asarray(result)
      counts["frames"] += len(output)
      counts["opposed"] += np.sum(output * wanted < 0)
      counts["opposed8"] += np.sum((output * wanted < 0) & (abs(output) >= 8) & (abs(wanted) >= 8))
      counts["under_by_6"] += np.sum((abs(wanted) >= 32) & ((wanted-output)*np.sign(wanted) > 5))
      counts["wire_within_1"] += np.sum(abs(output-recorded) <= 1)
      if end < len(wire) and state[end, 3] == 1 and times[end] - times[end-1] <= .05:
        counts["overrides"] += 1
        counts["pre_override_abs_sum"] += abs(output[-1])
        counts["pre_override_abs_max"] = max(counts["pre_override_abs_max"], abs(output[-1]))
      for threshold in (1, 12, 24, 36):
        hits = np.flatnonzero(abs(output) >= threshold)
        close = np.diff(times[start:end][hits]) < 1
        counts[threshold] += np.sum((output[hits[1:]] * output[hits[:-1]] < 0) & close)


def score_lane(data):
  model = data["model"]
  t = model[:, 0]
  control = latest(data["control"], t)
  corr = latest(data["correction"], t)[:, 1]
  state = latest(data["state"], t)
  cmd = latest(data["command"], t)
  valid = ((model[:, 6] >= .8) & (model[:, 5] >= 2.7) & (model[:, 5] <= 4.4)
           & (model[:, 7] == 0) & (model[:, 8] == 1) & (state[:, 1] > 9)
           & (state[:, 2] == 1) & (state[:, 3] == 0) & (state[:, 4] == 0)
           & (cmd[:, 1] == 1) & np.isfinite(corr + control[:, 1] + model[:, 2]))
  for start, stop in intervals(valid):
    while start < stop:
      end = np.searchsorted(t, t[start] + 8)
      if end > stop or t[end-1] - t[start] < 7.8:
        break
      s = slice(start, end)
      if np.max(np.diff(t[s])) < .12:
        demand = control[s, 1] * state[s, 1]**2
        half = demand - .5 * corr[s] * state[s, 1]**2
        def residual(values):
          return values - np.polyval(np.polyfit(t[s] - t[start], values, 1), t[s] - t[start])
        if np.std(residual(demand)) > .005:
          yield t[start], np.ptp(residual(model[s, 2])), np.std(residual(half))/np.std(residual(demand))
      start = end


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("directory", type=Path)
  files = sorted(parser.parse_args().directory.glob("*.npz"))
  if not files:
    parser.error("expected numbered NPZ segment extracts")
  for part in (0, 1):
    scores = {v: Counter() for v in ("baseline", "release4", "attenuated", "narrow")}
    windows = []
    for file in files:
      if (int(file.stem) >= 25) != part:
        continue
      with np.load(file) as data:
        score_torque(data, scores)
        windows.extend(score_lane(data))
    print("Route half", part + 1)
    for name, counts in scores.items():
      print(name, {k: int(v) for k, v in counts.items()})
    if windows:
      print("Lane windows, median excursion (m), median half-gain RMS ratio, worst window (s, m, ratio)")
      print(len(windows), *np.median(np.asarray(windows)[:, 1:], axis=0), max(windows, key=lambda w: w[2]))


if __name__ == "__main__":
  main()
