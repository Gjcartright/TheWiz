import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";


export async function captureTranscriptBatch({
  tab,
  queuePath,
  vaultDir,
  start = 0,
  batchSize = 20,
}) {
  const queue = JSON.parse(await fs.readFile(queuePath, "utf8"));
  const end = Math.min(queue.length, start + batchSize);
  const counts = { captured: 0, unavailable: 0, failed: 0, skipped: 0 };

  for (let index = start; index < end; index += 1) {
    const item = queue[index];
    const textPath = path.join(vaultDir, `${item.lecture_id}.txt`);
    const metadataPath = path.join(vaultDir, `${item.lecture_id}.metadata.json`);
    try {
      const existing = await readJson(metadataPath);
      if (
        existing.status === "captured" &&
        item.vault_status !== "blocked_duplicate_transcript_body"
      ) {
        counts.skipped += 1;
        continue;
      }
      const priorSidebarButton = tab.playwright.getByRole("button", {
        name: "Transcript in sidebar region",
        exact: true,
      });
      if (await priorSidebarButton.isVisible().catch(() => false)) {
        const priorExpanded = await priorSidebarButton
          .getAttribute("aria-expanded", { timeoutMs: 3000 })
          .catch(() => null);
        if (priorExpanded === "true") {
          await priorSidebarButton.click({ timeoutMs: 10000 }).catch(() => {});
          await tab.playwright.waitForTimeout(250);
        }
      }
      await tab.goto(item.lecture_url);
      await tab.playwright
        .waitForLoadState({ state: "domcontentloaded", timeoutMs: 30000 })
        .catch(() => {});
      await tab.playwright.waitForTimeout(800);
      const finalUrl = String((await tab.url()) || "");
      if (!finalUrl.includes("/learn/lecture/") || finalUrl.includes("/join/")) {
        await writeMetadata(metadataPath, item, {
          status: "capture_failed",
          blocker: "udemy_authentication_required",
          final_url: finalUrl,
        });
        counts.failed += 1;
        continue;
      }
      const panel = tab.playwright.getByRole("tabpanel", {
        name: "Transcript",
        exact: true,
      });
      let panelReady = await panel.isVisible().catch(() => false);
      if (!panelReady) {
        const sidebarButton = tab.playwright.getByRole("button", {
          name: "Transcript in sidebar region",
          exact: true,
        });
        await sidebarButton
          .waitFor({ state: "visible", timeoutMs: 8000 })
          .catch(() => {});
        if (await sidebarButton.isVisible().catch(() => false)) {
          const expanded = await sidebarButton
            .getAttribute("aria-expanded", { timeoutMs: 3000 })
            .catch(() => null);
          if (expanded !== "true") {
            await sidebarButton.click({ timeoutMs: 10000 }).catch(() => {});
          }
        } else {
          const transcriptTab = tab.playwright.getByRole("tab", {
            name: "Transcript",
            exact: true,
          });
          await transcriptTab
            .waitFor({ state: "visible", timeoutMs: 4000 })
            .catch(() => {});
          if (await transcriptTab.isVisible().catch(() => false)) {
            await transcriptTab.click({ timeoutMs: 10000 }).catch(() => {});
          }
        }
        await panel.waitFor({ state: "visible", timeoutMs: 10000 }).catch(() => {});
        panelReady = await panel.isVisible().catch(() => false);
      }
      if (panelReady) {
        const rawBody = await panel.innerText({ timeoutMs: 15000 }).catch(() => "");
        const cues = String(rawBody)
          .split(/\n\s*\n/)
          .map((value) => value.trim())
          .filter((value) => value && value !== "Autoscroll");
        const body = cues.join("\n\n").trim();
        if (body.length > 30 && cues.length) {
          const duplicateLectureId = await findDuplicateTranscriptId(
            vaultDir,
            item.lecture_id,
            body,
          );
          if (duplicateLectureId) {
            await fs.unlink(textPath).catch(() => {});
            await writeMetadata(metadataPath, item, {
              status: "transcript_unavailable",
              cue_count: 0,
              blocker: `udemy_duplicate_transcript_body_from_other_lecture:${duplicateLectureId}`,
            });
            counts.unavailable += 1;
            continue;
          }
          await fs.writeFile(textPath, `${body}\n`, "utf8");
          await writeMetadata(metadataPath, item, {
            status: "captured",
            cue_count: cues.length,
          });
          counts.captured += 1;
          continue;
        }
      }
      await writeMetadata(metadataPath, item, {
        status: "transcript_unavailable",
        cue_count: 0,
        blocker: "udemy_transcript_tab_or_text_not_available",
      });
      counts.unavailable += 1;
    } catch (error) {
      await writeMetadata(metadataPath, item, {
        status: "capture_failed",
        blocker: String(error?.message || error),
      }).catch(() => {});
      counts.failed += 1;
    }
  }
  return { start, end, total: queue.length, ...counts };
}


async function findDuplicateTranscriptId(vaultDir, lectureId, body) {
  const expectedHash = createHash("sha256").update(`${body}\n`).digest("hex");
  const entries = await fs.readdir(vaultDir, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".txt")) {
      continue;
    }
    const candidateId = entry.name.slice(0, -4);
    if (candidateId === lectureId) {
      continue;
    }
    const candidate = await fs.readFile(path.join(vaultDir, entry.name));
    const candidateHash = createHash("sha256").update(candidate).digest("hex");
    if (candidateHash === expectedHash) {
      return candidateId;
    }
  }
  return "";
}


async function readJson(filePath) {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf8"));
  } catch {
    return {};
  }
}


async function writeMetadata(filePath, item, values) {
  const payload = {
    schema_version: "udemy_transcript_capture.v1",
    lecture_id: item.lecture_id,
    lecture_url: item.lecture_url,
    course: item.course,
    section: item.section,
    video_title: item.video_title,
    captured_at_utc: new Date().toISOString(),
    transcript_text_ingested: false,
    live_signal_eligible: false,
    promotion_authority: "none_research_only",
    ...values,
  };
  await fs.writeFile(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}
