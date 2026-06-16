# Boids and Emergent Behavior: Week of 2026-06-15
Mustafa Nazeer and Joel Pampam, mentored by Dr. Diego Patino

(Figures and code were produced with AI assistance, disclosed per course policy.)

---

## Slide 1: Problem and Background

- Emergent group behavior: simple local rules produce complex global patterns.
- Reynolds boids (1987): three rules (separation, alignment, cohesion) are sufficient to
  produce realistic flocking without any centralized control.
- Why it matters here: this is exactly the behavior a Graph Neural Network will later learn
  to infer from agent trajectories, so we need a clean simulation as a ground truth source.

---

## Slide 2: What We Built

- Pure Python simulation: 120 boids on a 60x60 toroidal world, timestep 0.2 s.
- Three classic rules (separation, alignment, cohesion) over local neighborhoods.
- Obstacle avoidance around two circular obstacles.
- A pursuing predator that triggers collective escape behavior.
- Three quantitative metrics recorded every frame: polarization, mean nearest
  neighbor distance, and cluster count.
- Headless rendering via Matplotlib Agg backend (no display required).

---

## Slide 3: Flock Formation (Before Predator)

- See: `snapshot_early.png`, `snapshot_mid.png`
- Starting from random positions and velocities, the flock organizes itself within
  roughly 40 frames into a coherent moving group.
- Measured polarization at rest: approximately 1.00 (range 0 to 1; 1 means perfectly
  aligned). The flock moves as a single, coordinated group.
- Measured cluster count at rest: a single cluster, with visible spacing between
  individuals (mean nearest neighbor distance approximately 0.6 units), so the group
  reads as a flock of distinct boids rather than a collapsed blob.

---

## Slide 4: Predator Escape and Flock Splitting

- See: `flock_escape.gif`, `snapshot_late.png`
- When the predator engages, the flock shows a clear collective response.
- Cluster count spikes to approximately 7 (from 1): the flock splits into smaller
  fragments as boids flee in different directions.
- Polarization drops sharply toward 0: heading alignment collapses as individuals flee
  along different vectors, and mean nearest neighbor distance expands to roughly 3 units
  as the group scatters.
- This is the emergent event we set out to demonstrate: a single external perturbation
  (the predator) producing measurable, collective change in group structure.
- After the predator passes, the scattered boids stream back together into a cohesive
  fleeing group (visible in `snapshot_late.png`), so the escape, splitting, and partial
  regrouping are all observable. The predator keeps pursuing to the end of the run, so
  watch `flock_escape.gif` for the full splitting and regrouping dynamics.

---

## Slide 5: Metric Plots

- See: `polarization.png`, `nn_distance.png`, `cluster_count.png`
- Polarization plot: shows the settling of alignment as the flock forms, then the sharp
  drop when the predator engages.
- Nearest neighbor distance plot: spacing tightens during cohesion and spikes outward
  during the escape split.
- Cluster count plot: the rise from approximately 2 to approximately 12 during the escape
  gives a quantitative fingerprint of flock disruption.
- Together, these three metrics give us a signature we can use later to train or evaluate
  a GNN: a model that can reproduce this signature from raw trajectories has learned
  something real about collective behavior.

---

## Slide 6: Next Steps

- Weight tuning is complete for a clear flock formation and predator escape effect.
- Possible 3D extension: the simulation math is dimension agnostic (all rules operate on
  (N, D) arrays), so extending to D=3 is straightforward. Running a 3D version in Isaac
  Sim on a GPU machine is a stretch goal.
- Identify two real flocking datasets (for example, starling murmuration or fish school
  data) to validate qualitative behavior against.
- Begin framing the GNN architecture that will learn the boid interaction rules from
  trajectory data, so the network can reproduce emergent flocking without being told
  the rules explicitly.

---

## Joel's Contributions

[Mustafa to fill in with what Joel built and contributed, so credit is accurate and
specific. Do not leave this placeholder in the final version sent to Dr. Patino.]

---

*Draft prepared for weekly mentor meeting with Dr. Diego Patino. Finalize in your own
voice before presenting.*
