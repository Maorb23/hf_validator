# Infiro homepage design prototypes

Two isolated frontend directions for the provisional Infiro brand. No Python services, credentials, GPU operations, or live measurements are used.

## View locally

```bash
cd design-prototypes
npm ci
npm run dev
```

- Dark technical: <http://localhost:4317/dark>
- Light technical: <http://localhost:4317/light>

The root route opens the dark direction. The floating comparison control switches between directions. Both share content, architecture, and behavior so the visual treatment can be compared directly.

## Interactions

- Switch between the conceptual Pareto frontier and search landscape.
- Select latency, balanced, or throughput preference to highlight a different frontier point.
- Select architecture components or advance the walkthrough to inspect their inputs, outputs, and limitations.
- Switch inference engines to inspect illustrative search dimensions.
- Open measurement methodology; dismiss with Escape or the close button.
- Use the responsive mobile navigation.

All chart data is conceptual. Engine parameter names illustrate configuration dimensions, not verified model/image compatibility or runnable requests. Architecture selection never submits a job.

## Build and capture

```bash
npm run build
npx playwright install chromium
npm run dev
```

In a second terminal:

```bash
npm run capture
```

The capture script verifies key interactions, browser errors, and horizontal overflow, then writes full-page and hero screenshots for both directions at desktop (1440×1000) and mobile (390×844) sizes. Generated screenshots and build outputs are ignored by Git.

If using a custom browser directory, set `PLAYWRIGHT_BROWSERS_PATH` identically for browser installation and capture. `PROTOTYPE_URL` can point the capture script at another local port.

No contact or signup action is shown because no destination has been supplied. The final conversion concept remains “Discuss your workload” once a real address or scheduling link is available. These pages are design studies, not the full website or future operational console.
