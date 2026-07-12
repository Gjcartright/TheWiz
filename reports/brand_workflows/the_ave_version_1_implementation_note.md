# The Ave Version 1 Implementation Note

Version 1 has started as a standalone prototype at `apps/the-ave`.

## Current Build

- Full-screen The Ave world scene.
- `PRESS PLAY` entry with world controls hidden until entry.
- Five zones:
  - Sneaker District
  - Fashion Avenue
  - Producer Alley
  - Graffiti Tunnel
  - Block Party
- Store entrances after PLAY.
- Rich mock store identities across all five zones.
- Mock `Shoe Store #1` with the full commerce path.
- Featured mock products in Fashion Avenue, Producer Alley, Graffiti Tunnel, and Block Party.
- Mock product quick view.
- Size selection.
- Mock add-to-cart event.
- Canvas polish: parallax marks, signs, gates, posters, crates, speakers, subway marker, stage lights, and mural details.
- Mobile full-screen framing.
- Dialog semantics, Escape close behavior, focus return, accessible buttons, and zone fallback navigation.

## Gap-Fix Pass

The V1 gap-fix pass keeps the prototype static and no-build under `apps/the-ave`.
Mock data now lives in `apps/the-ave/data.js` as `window.TheAveData` with `zones`,
`stores`, `products`, and `integrations`. The main script uses lookup helpers so
the V2 Shopify replacement can swap catalog and checkout sources without changing
the world interaction model.

`Shoe Store #1` is the pilot mock Shopify path. Other stores are represented as
planned Shopify-connected merchants so The Ave reads as a populated retail world,
not one store surrounded by placeholders.

## Version Boundary

This is not connected to Shopify yet. Version 1 proves the world and store interaction model. Version 2 should connect `Shoe Store #1` to a real Shopify catalog, variants, cart, and checkout handoff, then expand that merchant connection model to additional stores.

## Local Run Command

```sh
python3 -m http.server 5173 --directory apps/the-ave
```

Local URL:

```text
http://localhost:5173
```

## Verification Completed

- JavaScript syntax check passed with `node --check apps/the-ave/script.js`.
- Desktop browser pass confirmed PLAY, zone entry, store entry, product quick view, size buttons, and mock cart button.
- Mobile viewport pass at 390x844 confirmed full-screen framing and in-viewport hotspots.
