# The Ave MCP Sequential Thinking Protocol

The Ave build work will use MCP Sequential Thinking as a required decision checkpoint.

## When To Use It

- Before choosing or changing the app architecture.
- Before starting a major implementation phase.
- Before defining the world/store/Shopify integration data model.
- Before making tradeoffs between 3D immersion and commerce usability.
- When a blocker, ambiguity, or risky assumption appears.
- Before final verification and handoff.

## How It Will Be Used

- Use Sequential Thinking to reason through the decision boundary.
- Record the resulting decision in normal project notes, plans, code, or final summaries.
- Keep implementation practical: small mechanical edits do not need their own separate thinking pass unless they affect architecture or product behavior.
- Keep The Ave model clear: the world is the platform, and Shopify is a per-store commerce integration.

## Current Build Hypothesis

Build The Ave as a custom virtual retail world with a multi-store system. Start with a playable world prototype, then add store profiles and slots, then connect one pilot Shopify store, then generalize the integration for multiple merchants.

## Verification Rule

Before calling a phase complete, verify:

- The world still feels immersive.
- Store discovery is clear.
- A store can connect to Shopify without making the whole project a Shopify storefront.
- Product browsing and checkout handoff remain easy.
- Mobile, fallback, accessibility, and performance are accounted for.
