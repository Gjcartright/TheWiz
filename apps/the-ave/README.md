# The Ave Version 1 Prototype

The Ave is a virtual retail world. Shopify is a per-store commerce connection that begins in Version 2.

This Version 1 prototype proves:

- Full-screen `THE AVE` world entry.
- `PRESS PLAY` interaction.
- Five world zones.
- Store entrances inside the world.
- One mock `Shoe Store #1`.
- Mock product quick view and size selection.
- Accessible zone/store controls.
- Mobile full-screen framing.

## Run Locally

From the repo root:

```sh
python3 -m http.server 5173 --directory apps/the-ave
```

Then open:

```text
http://localhost:5173
```

## Version 1 Boundary

This version intentionally does not connect to Shopify. The goal is to prove the world and shopping interaction model before building the multi-store commerce platform.
