# Bundled PiperX asset

The source model is from `agilexrobotics/piper_isaac_sim`, commit
`8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2`.

This submission includes only the URDF and meshes it references. Mesh paths are
relative to the URDF, and the collision geometry is stored as closed convex OBJ
meshes for Genesis.

The Link6 mass, centre of mass, and inertia are explicitly marked simulation
estimates rather than manufacturer measurements. They were integrated from the
closed components of `gripper_base_100_v2.dae` after applying the URDF scale and
mount transform, using a uniform effective density of 1500 kg/m³:

- mass: 0.299629179007 kg;
- centre of mass in Link6 coordinates:
  `(-0.000102303156, -0.000687033690, 0.033042637507)` m.

The complete calculation record is in
`urdf/genesis_asset_estimates.json`.
