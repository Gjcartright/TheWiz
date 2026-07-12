# The Ave Version 1 Gap Analysis

Reviewed: 2026-07-11
Local app: `apps/the-ave`
URL checked: `http://127.0.0.1:5173/`

## Executive Summary

Version 1 is on the right track. The prototype proves the core path: open The Ave, press PLAY, see zones, enter Sneaker District, enter `Shoe Store #1`, preview mock products, choose a size, and trigger a mock add-to-cart event.

The biggest gaps are not basic functionality. The gaps are depth, polish, and platform-readiness:

- The world currently feels like a strong illustrated prototype, not yet a premium Abeto-style world.
- Store discovery works, but only Sneaker District has a meaningful store/product experience.
- The UI still has visible website controls before PLAY, which weakens the game-title-screen feeling.
- Accessibility basics exist, but panels need stronger modal/focus behavior before public use.
- The data is still embedded in `script.js`, so Version 2 Shopify work needs cleaner store/product boundaries.

## Current Status Against Version 1 Criteria

| Criterion | Status | Notes |
|---|---:|---|
| Full-screen `THE AVE` opening scene | Pass | Full-screen canvas scene loads with brand title and world art. |
| `PRESS PLAY` entry action | Pass | Entry action works and reveals zones. |
| Five explorable zones | Pass | All five zones exist as visible buttons/hotspots. |
| Sample store entrances inside each zone | Partial | Each zone has at least one store, but only Sneaker District feels built out. |
| Detailed `Shoe Store #1` | Pass | Store panel, status label, and mock product cards exist. |
| Mock product quick view | Pass | Product detail, price, lore, size selection, and mock add-to-cart exist. |
| Mobile full-screen framing | Pass | 390x844 check kept hotspots visible and no page scrolling. |
| Reduced-motion/static fallback | Partial | CSS reduced-motion exists, but there is no separate static world fallback mode. |
| Keyboard-accessible zone/store controls | Partial | Controls are buttons/links, but panel focus management is not complete. |
| Basic analytics event stubs | Pass | Events log to console. Needs structured event adapter later. |

## Priority Gaps

### P1: Pre-Entry Chrome Weakens The World Feeling

Current issue: `Map`, `Stores`, and `Sound Off` are visible before the user presses PLAY.

Why it matters: The reference site feels like a game title screen because almost nothing competes with the world, title, and BEGIN action. The Ave should do the same.

Recommendation:

- Hide Map and Stores before PLAY.
- Keep only a small sound state if needed, or move sound control after entry.
- Make the opening screen focus on `THE AVE`, the world, and `PRESS PLAY`.

### P1: World Depth Is Still Prototype-Level

Current issue: The canvas world has a good direction, but it is still a flat 2D illustration. It does not yet have the density, dimensionality, or environmental storytelling of the reference.

Why it matters: The Ave needs to feel like a place merchants want to be inside, not only a homepage with hotspots.

Recommendation:

- Add more environmental details: street signs, storefront gates, sneaker boxes, posters, murals, speakers, stage lights, subway markers, crates.
- Add stronger dimensionality: parallax layers, foreground elements, shadow depth, animated storefront signs.
- For Version 1.5 or Version 2, decide whether to stay with optimized 2D/canvas or move toward Three.js/WebGL.

### P1: Store Discovery Is Too Uneven Across Zones

Current issue: Sneaker District has `Shoe Store #1` and mock products. Other zones mostly say coming soon.

Why it matters: Version 1 needs to prove the world contains stores, not just one store.

Recommendation:

- Add at least one believable mock store card per zone with distinct identity.
- Add one featured mock product to at least three zones.
- Keep `Shoe Store #1` as the only full mock commerce path if scope needs to stay lean.

### P1: Data Model Is Embedded In The UI Script

Current issue: zones, stores, and products live directly inside `script.js`.

Why it matters: Version 2 needs to swap mock products for Shopify-backed store data. That will be harder if the UI and data model stay coupled.

Recommendation:

- Split mock data into `data.js`.
- Shape the mock data like the future backend response:
  - `zones`
  - `storeSlots`
  - `stores`
  - `products`
  - `integrations`
- Add a thin data access layer such as `getZone`, `getStore`, `getProductsForStore`.

### P1: Panels Need Stronger Accessibility Behavior

Current issue: panels are visible/hidden and use buttons, but they do not behave like accessible modal/dialog surfaces.

Why it matters: The canvas is decorative, so the DOM controls must carry accessibility well.

Recommendation:

- Add `role="dialog"` and `aria-modal` where appropriate.
- Move focus into opened panels.
- Return focus to the triggering hotspot when a panel closes.
- Support Escape to close the current panel.
- Make close buttons visually and semantically consistent.

### P2: Store Panels Feel Like App Drawers, Not Store Interiors

Current issue: store and product surfaces are useful, but they appear as generic panels over the world.

Why it matters: The goal is an in-world retail experience. Product browsing should feel like entering a store, not opening a dashboard drawer.

Recommendation:

- Give each store a branded interior treatment.
- Use store-specific colors, sign, rack/shelf visual, and featured placement.
- Make `Shoe Store #1` feel like a boutique inside Sneaker District.

### P2: Analytics Are Console Logs Only

Current issue: tracking events are `console.info` calls.

Why it matters: This is okay for Version 1, but Version 2 needs real event collection.

Recommendation:

- Keep the same event names.
- Add an analytics adapter with a mock provider.
- Later swap the provider for a backend endpoint.

### P2: Version 2 Shopify Boundaries Need To Be More Visible

Current issue: the prototype says Shopify is mocked, but it does not yet show the store-level boundary strongly enough.

Why it matters: Visitors and merchants need to understand that each store can connect its own Shopify catalog.

Recommendation:

- Add store-level commerce labels:
  - `Connected store`
  - `Catalog source: Shopify in Version 2`
  - `Checkout: merchant Shopify checkout`
- Add a placeholder integration state to store cards.

### P2: No Automated UI Regression Check Yet

Current issue: verification was manual/browser-driven.

Why it matters: This visual prototype can regress easily with layout changes.

Recommendation:

- Add a simple smoke test script or Playwright test later.
- Required smoke path:
  - page loads
  - PLAY reveals zones
  - Sneaker District opens
  - Shoe Store #1 opens
  - Late Train Low product opens
  - mobile viewport keeps hotspots visible

## Suggested Next Fix Batch

Before calling Version 1 complete, do this batch:

1. Hide Map/Stores before PLAY.
2. Split data out of `script.js`.
3. Add richer mock store identities across all five zones.
4. Add focus/Escape handling for panels.
5. Add more world detail and at least two animated storefront accents.
6. Add a short `V1_READY_CHECKLIST.md` inside `apps/the-ave`.

## Version 1 Completion Definition

Version 1 is complete when:

- The first screen feels like a polished title screen.
- Pressing PLAY clearly reveals a world of stores.
- All five zones have believable store presence.
- `Shoe Store #1` demonstrates the full mock product path.
- Desktop and mobile framing are stable.
- Keyboard users can complete the same core path.
- The mock data shape is ready to be replaced by Version 2 Shopify data.

## Do Not Pull Into Version 1

These should stay in Version 2 or later:

- Real Shopify authentication.
- Real Shopify cart.
- Real checkout.
- Merchant self-service onboarding.
- Multi-tenant credential storage.
- Paid store placement.
- Multiplayer or avatars.

## Sequential Thinking Result

MCP Sequential Thinking checkpoint conclusion: The Ave Version 1 should remain focused on proving the world and store-discovery loop. The highest-value next work is not Shopify yet; it is making the world feel more intentional, making all zones feel populated, and separating mock store data so Version 2 can connect Shopify cleanly.
