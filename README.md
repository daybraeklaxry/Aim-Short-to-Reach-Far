<h1 align="center">Aim Short to Reach Far</h1>

<p align="center"><strong>Goal Distance Underestimates Your Frozen World Model</strong></p>

<p align="center">
  Xvyuan Liu, Jianjie Fang, Chen Gao, Yong Li<br>
  Tsinghua University
</p>

<p align="center">
  <a href="https://daybraeklaxry.github.io/Aim-Short-to-Reach-Far/"><strong>Project Page</strong></a>
  &emsp;
  <a href="https://daybraeklaxry.github.io/Aim-Short-to-Reach-Far/paper.pdf"><strong>Paper</strong></a>
  &emsp;
  <a href="https://daybraeklaxry.github.io/Aim-Short-to-Reach-Far/#demos"><strong>Demos</strong></a>
  &emsp;
  <a href="REPRODUCING.md"><strong>Code &amp; Data Guide</strong></a>
</p>

<p align="center">
  <a href="https://daybraeklaxry.github.io/Aim-Short-to-Reach-Far/">
    <img src="docs/assets/figure-1.svg" width="100%" alt="Anchored Planning: the same predictions can select different actions when the target changes.">
  </a>
</p>

**A frozen world model can support better control when the planning target changes.** Reaching a distant goal may require first moving away from it. Anchored Planning retrieves a recorded observation as an intermediate target, then uses the frozen model to evaluate actions from the current state. The same target supports **AP-CEM**, which synthesizes actions, and **AP-rank**, which selects among recorded action blocks.

## Demos

<table>
  <tr>
    <th align="center">AP-rank: select actions</th>
    <th align="center">AP-CEM: synthesize actions</th>
  </tr>
  <tr>
    <td width="50%" align="center">
      <a href="https://daybraeklaxry.github.io/Aim-Short-to-Reach-Far/assets/demos/rank-pusht-001.mp4">
        <img src="docs/assets/demos/rank-pusht-001-poster.jpg" alt="PushT after a start perturbation: Direct fails while AP-rank reaches the goal." width="420">
      </a>
    </td>
    <td width="50%" align="center">
      <a href="https://daybraeklaxry.github.io/Aim-Short-to-Reach-Far/assets/demos/pusht-000.mp4">
        <img src="docs/assets/demos/pusht-000-poster.jpg" alt="PushT with the same frozen predictor: final-goal CEM fails while AP-CEM reaches the goal." width="420">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">Choose actions for the current state after a perturbation.<br><a href="https://daybraeklaxry.github.io/Aim-Short-to-Reach-Far/assets/demos/rank-pusht-001.mp4">Watch AP-rank</a></td>
    <td align="center">Search toward an observed target with the same frozen model.<br><a href="https://daybraeklaxry.github.io/Aim-Short-to-Reach-Far/assets/demos/pusht-000.mp4">Watch AP-CEM</a></td>
  </tr>
</table>

These are selected successful cases from paired evaluations. [Explore all twelve demos](https://daybraeklaxry.github.io/Aim-Short-to-Reach-Far/#demos) across Cube, PushT, Reacher, and TwoRoom, or see the [full evaluation results](https://daybraeklaxry.github.io/Aim-Short-to-Reach-Far/#results).

## Using the repository

- **[Code and setup](code/ap/README.md#external-assets):** controller implementation, pretrained models, datasets, and simulator dependencies.
- **[Reproduction guide](REPRODUCING.md):** recompute the paper's tables, run evaluations, and render paired demonstrations from saved states.
- **[Recorded results](data/README.md):** per-episode measurements, definitions, and units.
- **[LeWM evaluation](lewm_checks/README.md):** released-planner settings and comparisons with the released evaluator.

The [reproduction guide](REPRODUCING.md#code-and-data-layout) maps each experiment to its code and data.

## Citation

```bibtex
@misc{liu2026aimshort,
  title  = {Aim Short to Reach Far: Goal Distance Underestimates Your Frozen World Model},
  author = {Xvyuan Liu and Jianjie Fang and Chen Gao and Yong Li},
  year   = {2026},
  url    = {https://github.com/daybraeklaxry/Aim-Short-to-Reach-Far}
}
```

## License

Original code is released under the [MIT License](LICENSE). Parts derived from LeWM retain their original license and notices in [`code/ap/LICENSE`](code/ap/LICENSE) and [`code/ap/NOTICE.md`](code/ap/NOTICE.md). External datasets, simulators, and checkpoints remain subject to their original terms and are not redistributed here.
