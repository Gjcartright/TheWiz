# The Ave V1 Ready Checklist

## V1 Scope

- [x] Keep The Ave as a static no-build 2D/canvas prototype.
- [x] Keep Shopify mocked in Version 1.
- [x] Treat The Ave as the virtual world, not as a Shopify storefront.
- [x] Model each store as a future merchant commerce connection.

## World Entry

- [x] Opening screen focuses on the world art, `THE AVE`, and `PRESS PLAY`.
- [x] Hide `Map`, `Stores`, and sound controls before entry.
- [x] Reveal world controls and zone hotspots after `PRESS PLAY`.

## World Depth

- [x] Add canvas parallax details behind the world.
- [x] Add street signs, storefront gates, sneaker boxes, posters, speaker stacks, subway markers, crates, stage lights, and mural marks.
- [x] Add animated sign/gate shimmer accents on storefronts.

## Store Discovery

- [x] Five zones are populated with believable store identities.
- [x] `Shoe Store #1` remains the complete mock commerce path.
- [x] Fashion Avenue, Producer Alley, and Graffiti Tunnel each include featured mock products.
- [x] Block Party includes featured mock launch/event stores.

## Data Readiness

- [x] `data.js` exposes `window.TheAveData`.
- [x] Data is organized as `zones`, `stores`, `products`, and `integrations`.
- [x] `script.js` reads through lookup helpers for zones, stores, products, and products by store.
- [x] Data remains shaped for future Shopify replacement without real Shopify calls.

## Accessibility

- [x] Zone, store, and product panels use dialog semantics.
- [x] Focus moves into opened panels.
- [x] Close actions return focus to the trigger.
- [x] Escape closes the active panel.
- [x] Product quick view close/Escape returns to the store panel.
- [x] Zone, store, and product actions remain reachable through real DOM buttons.

## Verification

- [x] `node --check apps/the-ave/data.js`
- [x] `node --check apps/the-ave/script.js`
- [x] Opening screen exposes only `PRESS PLAY` before entry.
- [x] `PRESS PLAY` reveals controls and all five zone hotspots.
- [x] All five zones open with populated store cards.
- [x] `Shoe Store #1` opens the complete mock commerce path.
- [x] `Late Train Low` quick view opens, size selection works, and mock add-to-cart confirms.
- [x] Mobile viewport `390x844` fits without page overflow.
- [x] Required events fire: `play_pressed`, `zone_entered`, `store_entered`, `product_quick_viewed`, and `track_quick_add`.

## Version 2 Handoff

- [ ] Replace selected mock store products with Shopify Storefront API catalog data.
- [ ] Map products to Shopify product handles and variant IDs.
- [ ] Connect selected size/variant to a merchant-owned checkout handoff.
- [ ] Add merchant onboarding and store connection management.
- [ ] Decide whether the world should stay 2D/canvas or move to Three.js/WebGL.
