const worldData = window.TheAveData;

if (!worldData) {
  throw new Error("The Ave data layer is missing. Load data.js before script.js.");
}

const state = {
  activeZone: null,
  activeStore: null,
  activeProduct: null,
  selectedSize: null,
  live: false,
  muted: true,
  time: 0,
  activePanel: null,
  lastTrigger: null,
};

const stage = document.querySelector(".world-stage");
const canvas = document.querySelector("#worldCanvas");
const ctx = canvas.getContext("2d");
const introPanel = document.querySelector("#introPanel");
const zoneLayer = document.querySelector("#zoneLayer");
const zonePanel = document.querySelector("#zonePanel");
const storePanel = document.querySelector("#storePanel");
const productPanel = document.querySelector("#productPanel");
const playButton = document.querySelector("#playButton");
const mapButton = document.querySelector("#mapButton");
const directoryButton = document.querySelector("#directoryButton");
const muteButton = document.querySelector("#muteButton");

const zoneTitle = document.querySelector("#zoneTitle");
const zoneEyebrow = document.querySelector("#zoneEyebrow");
const zoneDescription = document.querySelector("#zoneDescription");
const storeList = document.querySelector("#storeList");
const storeTitle = document.querySelector("#storeTitle");
const storeEyebrow = document.querySelector("#storeEyebrow");
const storeDescription = document.querySelector("#storeDescription");
const shopifyStatus = document.querySelector("#shopifyStatus");
const commerceMeta = document.querySelector("#commerceMeta");
const productGrid = document.querySelector("#productGrid");
const productStore = document.querySelector("#productStore");
const productTitle = document.querySelector("#productTitle");
const productPrice = document.querySelector("#productPrice");
const productLore = document.querySelector("#productLore");
const productArt = document.querySelector("#productArt");
const sizeOptions = document.querySelector("#sizeOptions");
const mockCartButton = document.querySelector("#mockCartButton");

function getZone(zoneId) {
  return worldData.zones.find((zone) => zone.id === zoneId) || null;
}

function getStore(storeId) {
  return worldData.stores.find((store) => store.id === storeId) || null;
}

function getProduct(productId) {
  return worldData.products.find((product) => product.id === productId) || null;
}

function getIntegration(integrationId) {
  return worldData.integrations.find((integration) => integration.id === integrationId) || null;
}

function getStoresForZone(zoneId) {
  const zone = getZone(zoneId);
  return zone ? zone.storeIds.map(getStore).filter(Boolean) : [];
}

function getProductsForStore(storeId) {
  const store = getStore(storeId);
  return store ? store.productIds.map(getProduct).filter(Boolean) : [];
}

function resizeCanvas() {
  const ratio = Math.min(window.devicePixelRatio || 1, 1.8);
  canvas.width = Math.floor(window.innerWidth * ratio);
  canvas.height = Math.floor(window.innerHeight * ratio);
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawWorld() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const centerX = width * 0.5;
  const centerY = height * (width < 760 ? 0.46 : 0.5);
  const scale = Math.min(width, height) / 760;
  const bob = Math.sin(state.time * 0.0012) * 8;

  ctx.clearRect(0, 0, width, height);
  drawClouds(width, height);
  drawParallaxTags(width, height);

  ctx.save();
  ctx.translate(centerX, centerY + bob);
  ctx.scale(scale, scale);
  ctx.rotate(Math.sin(state.time * 0.00035) * 0.018);

  drawIsland();
  drawRoads();
  drawDistricts();
  drawWorldDetails();
  drawLetterBlocks();
  drawStorefronts();
  drawForegroundDetails();

  ctx.restore();

  state.time += 16;
  requestAnimationFrame(drawWorld);
}

function drawClouds(width, height) {
  ctx.save();
  ctx.globalAlpha = 0.18;
  ctx.fillStyle = "#f4f1df";
  for (let index = 0; index < 8; index += 1) {
    const x = ((state.time * 0.01 + index * 240) % (width + 260)) - 160;
    const y = (index * 97) % height;
    ctx.beginPath();
    ctx.ellipse(x, y, 60, 18, 0, 0, Math.PI * 2);
    ctx.ellipse(x + 45, y + 8, 48, 14, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawParallaxTags(width, height) {
  const offset = Math.sin(state.time * 0.0005) * 18;
  ctx.save();
  ctx.globalAlpha = 0.12;
  ctx.strokeStyle = "#f4f1df";
  ctx.lineWidth = 3;
  for (let index = 0; index < 7; index += 1) {
    const x = (index * 217 + offset * (index % 3)) % width;
    const y = (index * 151 + offset) % height;
    ctx.strokeRect(x, y, 12, 12);
    ctx.beginPath();
    ctx.moveTo(x + 2, y + 8);
    ctx.lineTo(x + 10, y + 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawIsland() {
  ctx.fillStyle = "#3d7f63";
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(-245, -82);
  ctx.bezierCurveTo(-198, -238, 24, -275, 205, -175);
  ctx.bezierCurveTo(310, -118, 322, 94, 206, 192);
  ctx.bezierCurveTo(75, 304, -160, 245, -245, 108);
  ctx.bezierCurveTo(-310, 1, -285, -36, -245, -82);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#1f5f55";
  ctx.globalAlpha = 0.35;
  ctx.beginPath();
  ctx.ellipse(16, 214, 260, 45, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
}

function drawRoads() {
  ctx.strokeStyle = "#2a2e2d";
  ctx.lineWidth = 28;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(-210, 80);
  ctx.quadraticCurveTo(-40, 10, 185, 62);
  ctx.moveTo(-50, -190);
  ctx.quadraticCurveTo(20, -60, -18, 190);
  ctx.stroke();

  ctx.strokeStyle = "#f4f1df";
  ctx.lineWidth = 4;
  ctx.setLineDash([16, 18]);
  ctx.beginPath();
  ctx.moveTo(-210, 80);
  ctx.quadraticCurveTo(-40, 10, 185, 62);
  ctx.moveTo(-50, -190);
  ctx.quadraticCurveTo(20, -60, -18, 190);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawDistricts() {
  drawBlock(-175, 28, 118, 78, "#d94f3d");
  drawBlock(-52, -142, 132, 96, "#f2c94c");
  drawBlock(116, -40, 128, 92, "#684f9b");
  drawBlock(-18, 112, 150, 80, "#50c0b8");
  drawBlock(128, -176, 106, 84, "#6ea66d");
}

function drawBlock(x, y, width, height, color) {
  ctx.save();
  ctx.translate(x, y);
  ctx.fillStyle = color;
  ctx.strokeStyle = "#101111";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.roundRect(-width / 2, -height / 2, width, height, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "rgba(244, 241, 223, 0.38)";
  for (let row = 0; row < 2; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      ctx.fillRect(-width / 2 + 18 + col * 34, -height / 2 + 18 + row * 28, 18, 12);
    }
  }
  ctx.restore();
}

function drawWorldDetails() {
  drawSign(-232, 5, "KICKS", "#f2c94c");
  drawSign(-8, -190, "HOODS", "#d94f3d");
  drawSign(178, -80, "BEATS", "#50c0b8");
  drawPosterWall(64, 136);
  drawCrates(-128, 164);
  drawSpeakers(224, 20);
  drawSubwayMarker(-92, -18);
  drawStageLights(164, -202);
  drawMuralMarks(-12, 198);
}

function drawSign(x, y, text, color) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-0.08);
  ctx.fillStyle = color;
  ctx.strokeStyle = "#101111";
  ctx.lineWidth = 3;
  ctx.fillRect(-38, -18, 76, 36);
  ctx.strokeRect(-38, -18, 76, 36);
  ctx.fillStyle = "#101111";
  ctx.font = "900 17px Arial Black, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(text, 0, 6);
  ctx.restore();
}

function drawPosterWall(x, y) {
  ctx.save();
  ctx.translate(x, y);
  const colors = ["#f2c94c", "#d94f3d", "#f4f1df", "#684f9b"];
  colors.forEach((color, index) => {
    ctx.fillStyle = color;
    ctx.strokeStyle = "#101111";
    ctx.lineWidth = 2;
    ctx.fillRect(index * 20 - 42, (index % 2) * 8 - 20, 18, 28);
    ctx.strokeRect(index * 20 - 42, (index % 2) * 8 - 20, 18, 28);
  });
  ctx.restore();
}

function drawCrates(x, y) {
  ctx.save();
  ctx.translate(x, y);
  for (let index = 0; index < 4; index += 1) {
    ctx.fillStyle = index % 2 ? "#f2c94c" : "#d94f3d";
    ctx.strokeStyle = "#101111";
    ctx.lineWidth = 2;
    ctx.fillRect(index * 18, -index * 5, 22, 18);
    ctx.strokeRect(index * 18, -index * 5, 22, 18);
  }
  ctx.restore();
}

function drawSpeakers(x, y) {
  ctx.save();
  ctx.translate(x, y);
  ctx.fillStyle = "#101111";
  ctx.fillRect(-18, -36, 36, 72);
  ctx.fillRect(24, -24, 30, 60);
  ctx.fillStyle = "#f4f1df";
  [-18, 12].forEach((yPos) => {
    ctx.beginPath();
    ctx.arc(0, yPos, 10, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.beginPath();
  ctx.arc(39, 4, 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawSubwayMarker(x, y) {
  ctx.save();
  ctx.translate(x, y);
  ctx.fillStyle = "#f2c94c";
  ctx.strokeStyle = "#101111";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(0, 0, 18, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#101111";
  ctx.font = "900 18px Arial Black, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("A", 0, 7);
  ctx.restore();
}

function drawStageLights(x, y) {
  ctx.save();
  ctx.translate(x, y);
  const flicker = 0.35 + Math.sin(state.time * 0.012) * 0.18;
  ctx.globalAlpha = flicker;
  ctx.fillStyle = "#f2c94c";
  ctx.beginPath();
  ctx.moveTo(-24, 0);
  ctx.lineTo(0, 88);
  ctx.lineTo(42, 0);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#101111";
  ctx.fillRect(-28, -10, 70, 14);
  ctx.restore();
}

function drawMuralMarks(x, y) {
  ctx.save();
  ctx.translate(x, y);
  ctx.strokeStyle = "#f4f1df";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(-42, -8);
  ctx.bezierCurveTo(-22, -34, 10, 26, 42, -14);
  ctx.moveTo(-30, 18);
  ctx.lineTo(34, 20);
  ctx.stroke();
  ctx.restore();
}

function drawLetterBlocks() {
  ctx.save();
  ctx.translate(-8, -18);
  drawBigText("THE", -130, -30);
  drawBigText("AVE", -138, 58);
  ctx.restore();
}

function drawBigText(text, x, y) {
  ctx.save();
  ctx.font = "900 86px Impact, Arial Black, sans-serif";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#101111";
  ctx.lineWidth = 14;
  ctx.fillStyle = "#e9eadc";
  ctx.translate(x, y);
  ctx.strokeText(text, 0, 0);
  ctx.fillText(text, 0, 0);
  ctx.fillStyle = "rgba(16, 17, 17, 0.28)";
  ctx.fillText(text, 8, 10);
  ctx.restore();
}

function drawStorefronts() {
  const shimmer = Math.sin(state.time * 0.01) * 0.5 + 0.5;
  drawStorefront(-178, 92, "#f2c94c", "SHOP", shimmer);
  drawStorefront(-64, -78, "#f4f1df", "LAB", 1 - shimmer);
  drawStorefront(126, 26, "#f4f1df", "GEAR", shimmer);
  drawStorefront(20, 168, "#f4f1df", "ART", 1 - shimmer);
  drawStorefront(156, -120, "#f4f1df", "LIVE", shimmer);
}

function drawStorefront(x, y, color, label, shimmer) {
  ctx.save();
  ctx.translate(x, y);
  ctx.fillStyle = color;
  ctx.strokeStyle = "#101111";
  ctx.lineWidth = 3;
  ctx.fillRect(-28, -24, 56, 48);
  ctx.strokeRect(-28, -24, 56, 48);
  ctx.fillStyle = `rgba(242, 201, 76, ${0.25 + shimmer * 0.55})`;
  ctx.fillRect(-24, -20, 48, 10);
  ctx.fillStyle = "#101111";
  ctx.font = "900 9px Arial Black, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, 0, -11);
  ctx.fillRect(-20, -4, 40, 5);
  ctx.fillStyle = "#50c0b8";
  ctx.fillRect(-10, 6, 20, 18);
  ctx.strokeRect(-10, 6, 20, 18);
  ctx.restore();
}

function drawForegroundDetails() {
  ctx.save();
  ctx.globalAlpha = 0.72;
  ctx.fillStyle = "#101111";
  ctx.beginPath();
  ctx.ellipse(0, 222, 230, 34, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function enterWorld() {
  state.live = true;
  stage.classList.add("is-live");
  [mapButton, directoryButton, muteButton].forEach((button) => {
    button.disabled = false;
  });
  window.setTimeout(() => {
    introPanel.hidden = true;
    zoneLayer.hidden = false;
    mapButton.focus();
  }, 520);
  track("play_pressed");
}

function setActivePanel(panelName, panel, trigger) {
  [zonePanel, storePanel, productPanel].forEach((item) => {
    item.hidden = item !== panel;
  });
  state.activePanel = panelName;
  state.lastTrigger = trigger || state.lastTrigger || document.activeElement;
  panel.hidden = false;
  window.requestAnimationFrame(() => {
    const closeButton = panel.querySelector(".close-button");
    if (closeButton) closeButton.focus();
  });
}

function closeActivePanel({ restoreFocus = true } = {}) {
  const panelByName = {
    zone: zonePanel,
    store: storePanel,
    product: productPanel,
  };
  const panel = panelByName[state.activePanel];
  if (panel) panel.hidden = true;
  state.activePanel = null;
  if (restoreFocus && state.lastTrigger && typeof state.lastTrigger.focus === "function") {
    state.lastTrigger.focus();
  }
}

function closeProductToStore() {
  productPanel.hidden = true;
  state.activePanel = null;
  const store = getStore(state.activeStore);
  if (store) {
    openStore(store.id, state.lastTrigger);
  } else if (state.lastTrigger && typeof state.lastTrigger.focus === "function") {
    state.lastTrigger.focus();
  }
}

function openZone(zoneId, trigger = document.activeElement) {
  const zone = getZone(zoneId);
  if (!zone) return;
  state.activeZone = zoneId;
  zoneEyebrow.textContent = zone.eyebrow;
  zoneTitle.textContent = zone.title;
  zoneDescription.textContent = zone.description;
  storeList.innerHTML = "";
  getStoresForZone(zoneId).forEach((store) => {
    const integration = getIntegration(store.integrationId);
    const productCount = getProductsForStore(store.id).length;
    const button = document.createElement("button");
    button.className = "store-card";
    button.type = "button";
    button.dataset.storeId = store.id;
    button.style.setProperty("--store-accent", store.accent);
    button.innerHTML = `<strong>${store.name}</strong><span>${store.type} · ${productCount} mock product${productCount === 1 ? "" : "s"} · ${integration.label}</span><em>${store.description}</em>`;
    button.addEventListener("click", () => openStore(store.id, button));
    storeList.append(button);
  });
  setActivePanel("zone", zonePanel, trigger);
  track("zone_entered", { zone: zoneId });
}

function openStore(storeId, trigger = document.activeElement) {
  const store = getStore(storeId);
  if (!store) return;
  const integration = getIntegration(store.integrationId);
  const products = getProductsForStore(storeId);
  state.activeStore = storeId;
  storeEyebrow.textContent = `${integration.platform} ${integration.status}`;
  storeTitle.textContent = store.name;
  storeDescription.textContent = store.description;
  shopifyStatus.textContent = integration.label;
  commerceMeta.innerHTML = `<span>Catalog source: ${integration.catalogSource}</span><span>Checkout: ${integration.checkout}</span>`;
  productGrid.innerHTML = "";
  products.forEach((product) => {
    const button = document.createElement("button");
    button.className = "product-card";
    button.type = "button";
    button.dataset.productId = product.id;
    button.style.setProperty("--store-accent", store.accent);
    button.innerHTML = `<strong>${product.name}</strong><span>${product.price} · ${product.lore}</span>`;
    button.addEventListener("click", () => openProduct(product.id, button));
    productGrid.append(button);
  });
  setActivePanel("store", storePanel, trigger);
  track("store_entered", { store: storeId });
}

function openProduct(productId, trigger = document.activeElement) {
  const product = getProduct(productId);
  const store = worldData.stores.find((item) => item.productIds.includes(productId));
  if (!product || !store) return;
  state.activeProduct = productId;
  state.selectedSize = product.sizes[0];
  productStore.textContent = store.name;
  productTitle.textContent = product.name;
  productPrice.textContent = product.price;
  productLore.textContent = product.lore;
  productArt.style.setProperty("--product-color", product.color);
  sizeOptions.innerHTML = "";
  product.sizes.forEach((size) => {
    const button = document.createElement("button");
    button.className = `size-option${size === state.selectedSize ? " is-selected" : ""}`;
    button.type = "button";
    button.dataset.size = size;
    button.textContent = size;
    button.addEventListener("click", () => selectSize(size));
    sizeOptions.append(button);
  });
  setActivePanel("product", productPanel, trigger);
  track("product_quick_viewed", { store: store.id, product: product.id });
}

function selectSize(size) {
  state.selectedSize = size;
  [...sizeOptions.children].forEach((button) => {
    button.classList.toggle("is-selected", button.textContent === size);
  });
}

function track(eventName, payload = {}) {
  const eventRecord = { eventName, payload };
  const existingEvents = JSON.parse(document.documentElement.dataset.aveEvents || "[]");
  const nextEvents = [...existingEvents, eventRecord].slice(-20);
  document.documentElement.dataset.lastAveEvent = eventName;
  document.documentElement.dataset.aveEvents = JSON.stringify(nextEvents);
  window.dispatchEvent(new CustomEvent("the-ave-event", { detail: eventRecord }));
  console.info("[The Ave event]", eventName, payload);
}

playButton.addEventListener("click", enterWorld);
mapButton.addEventListener("click", () => {
  if (!state.live) return;
  openZone(state.activeZone || "sneaker", mapButton);
});
directoryButton.addEventListener("click", () => {
  if (!state.live) return;
  openZone("sneaker", directoryButton);
});
muteButton.addEventListener("click", () => {
  state.muted = !state.muted;
  muteButton.textContent = state.muted ? "Sound Off" : "Sound On";
  muteButton.setAttribute("aria-label", state.muted ? "Sound is off" : "Sound is on");
});

document.querySelectorAll("[data-zone]").forEach((button) => {
  button.addEventListener("click", () => openZone(button.dataset.zone, button));
});

document.querySelectorAll("[data-zone-link]").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    if (!state.live) enterWorld();
    openZone(link.dataset.zoneLink, link);
  });
});

document.querySelector("#closeZoneButton").addEventListener("click", () => closeActivePanel());
document.querySelector("#closeStoreButton").addEventListener("click", () => closeActivePanel());
document.querySelector("#closeProductButton").addEventListener("click", closeProductToStore);

mockCartButton.addEventListener("click", () => {
  const product = getProduct(state.activeProduct);
  const store = product
    ? worldData.stores.find((item) => item.productIds.includes(product.id))
    : null;
  if (!product || !store) return;
  track("track_quick_add", { store: store.id, product: product.id, size: state.selectedSize });
  mockCartButton.textContent = `Mock Added · Size ${state.selectedSize}`;
  window.setTimeout(() => {
    mockCartButton.textContent = "Mock Add To Cart";
  }, 1600);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.activePanel) {
    if (state.activePanel === "product") {
      closeProductToStore();
    } else {
      closeActivePanel();
    }
  }
});

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
drawWorld();
