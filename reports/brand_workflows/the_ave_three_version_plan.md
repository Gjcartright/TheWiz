# The Ave: Three-Version Build Plan

The Ave is a virtual retail world first. Shopify is a per-store commerce connection inside that world.

## Version 1: The Ave World Prototype

Goal: prove the immersive world, the brand feeling, and the store discovery loop before building the full commerce platform.

Estimated scope: 2-4 weeks.

### What It Includes

- Full-screen `THE AVE` opening scene.
- `PRESS PLAY` entry action.
- Five explorable zones:
  - Sneaker District
  - Fashion Avenue
  - Producer Alley
  - Graffiti Tunnel
  - Block Party
- Sample store entrances inside each zone.
- One detailed sample store: `Shoe Store #1`.
- Mock products for the sample store.
- Product quick-view drawer with image, name, price, sizes, lore, and disabled/mock add-to-cart.
- Mobile full-screen framing.
- Reduced-motion/static fallback.
- Keyboard-accessible zone and store buttons.
- Basic analytics event stubs:
  - `play_pressed`
  - `zone_entered`
  - `store_entered`
  - `product_quick_viewed`

### What It Does Not Include Yet

- Real Shopify connection.
- Real checkout.
- Merchant onboarding.
- Multi-store tenant management.
- Payment or billing for store owners.
- Advanced avatars or multiplayer.

### Acceptance Criteria

- The first screen feels like a virtual hip hop retail world within 3 seconds.
- A visitor can press PLAY and understand that The Ave contains stores.
- A visitor can enter `Shoe Store #1`.
- A visitor can preview mock products inside that store.
- The scene works on desktop and mobile without broken cropping.
- The world has accessible non-canvas buttons for all key actions.

### Main Risk

The prototype could become too much visual art and not enough retail clarity. The fix is to make stores visible immediately after PLAY.

## Version 2: Shopify-Connected Pilot

Goal: prove that one real store inside The Ave can connect to Shopify and sell products.

Estimated scope: 4-8 weeks after Version 1.

### What It Includes

- Backend API for stores, zones, products, and Shopify connection settings.
- Database for:
  - zones
  - store slots
  - store profiles
  - Shopify integration records
  - product mapping
  - analytics events
- One real pilot store connected to Shopify.
- Shopify Storefront API integration for:
  - products
  - collections
  - product images
  - variants
  - prices
  - availability
  - cart creation
  - checkout handoff
- Store detail view inside the world.
- Real product quick view.
- Variant and size selection.
- Add-to-cart for the pilot store.
- Checkout handoff to Shopify.
- Store-level search or product filtering.
- Basic admin seed/config file for the pilot store.
- Error states for disconnected Shopify, empty catalog, sold-out products, and checkout failure.

### What It Does Not Include Yet

- Public self-serve merchant onboarding.
- Many live Shopify stores.
- Store owner billing.
- Complex inventory sync beyond Storefront API reads.
- Full moderation workflow.

### Acceptance Criteria

- `Shoe Store #1` loads real products from its Shopify catalog.
- A visitor can select a variant and add it to cart.
- A visitor can proceed to that store's Shopify checkout.
- The world remains usable while product data loads.
- The visitor can clearly tell which store they are shopping from.
- Shopify credentials are not exposed in the frontend.
- The pilot integration can be repeated for another store without rewriting the world.

### Main Risk

The commerce flow could feel bolted on. The fix is to make product cards and checkout handoff feel like part of the store interior, while still using Shopify for the serious transaction work.

## Version 3: Multi-Store Platform

Goal: turn The Ave from a pilot experience into a repeatable virtual retail platform where multiple merchants can operate stores.

Estimated scope: 8-16 weeks after Version 2, depending on merchant onboarding depth.

### What It Includes

- Merchant/store owner onboarding.
- Shopify connection flow for each merchant.
- Store profile editor:
  - store name
  - logo
  - description
  - brand colors
  - zone preference
  - featured collections
  - featured products
  - store art
- Store slot management:
  - assign store to zone
  - set visibility
  - feature/promote stores
  - schedule drops
- Multi-store product browsing.
- Global world search:
  - stores
  - products
  - drops
  - zones
- Store analytics dashboard:
  - store visits
  - product views
  - add-to-cart events
  - checkout handoffs
  - top zones
  - top products
- Moderation/review before a store goes live.
- Drop/event system:
  - timed releases
  - hidden products
  - collab events
  - featured block party moments
- Monetization foundation:
  - free/pilot store
  - featured placement
  - monthly store placement
  - launch/drop promotion packages

### What It Does Not Include Yet

- Full multiplayer social layer unless it becomes necessary.
- In-world payments outside Shopify.
- Custom warehouse/fulfillment logic.
- Replacing Shopify as the merchant system of record.

### Acceptance Criteria

- At least 3 Shopify-connected stores can live in The Ave at once.
- Each store can manage its own profile and featured products.
- Store owners can connect or configure Shopify without code changes.
- Visitors can move between stores without confusion about which cart/store they are shopping.
- The Ave team can approve, feature, hide, or reposition stores.
- Analytics show which zones and stores are performing.
- The platform is ready to onboard real merchants beyond the first pilot.

### Main Risk

Multi-store commerce can become operationally messy. The fix is to keep clean boundaries: The Ave owns world placement and discovery; Shopify owns each store's catalog, cart, checkout, inventory, and fulfillment.

## Version Order

1. Build the feeling.
2. Prove one real store can sell.
3. Turn the store connection into a repeatable platform.

## Sequential Thinking Checkpoints

Use MCP Sequential Thinking before starting each version and before declaring each version complete.

- Version 1 checkpoint: world concept, interaction model, mobile framing.
- Version 2 checkpoint: Shopify integration architecture, cart/checkout flow, data security.
- Version 3 checkpoint: merchant onboarding, multi-tenant model, analytics, moderation, monetization.
