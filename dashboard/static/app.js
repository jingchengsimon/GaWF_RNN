const tasksNode = document.querySelector("#tasks");
const template = document.querySelector("#task-template");
const connection = document.querySelector("#connection");

function render(state) {
  tasksNode.replaceChildren();
  for (const task of state.tasks || []) {
    const card = template.content.firstElementChild.cloneNode(true);
    const percent = task.expected_total > 0
      ? Math.min(100, (task.valid_count / task.expected_total) * 100)
      : 0;
    card.querySelector("h2").textContent = task.description;
    card.querySelector(".count").textContent = `${task.valid_count} / ${task.expected_total}`;
    const jobIds = (task.job_ids || []).map((id) => `Job ${id}`).join(", ");
    card.querySelector(".task-meta").textContent = [task.machine, jobIds]
      .filter(Boolean)
      .join(" · ");
    const track = card.querySelector(".track");
    track.setAttribute("aria-valuenow", String(Math.round(percent)));
    card.querySelector(".fill").style.width = `${percent}%`;
    if (task.stale) card.classList.add("stale");
    tasksNode.append(card);
  }

  const offline = Object.entries(state.machines || {}).filter(([, item]) => !item.online);
  if (offline.length) {
    connection.textContent = "远程连接暂时不可用，当前显示上次确认的进度。";
    connection.classList.add("offline");
  } else if (state.updated_at) {
    connection.textContent = `自动刷新 · ${new Date(state.updated_at).toLocaleString()}`;
    connection.classList.remove("offline");
  }
}

async function refreshView() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (_) {
    connection.textContent = "进度服务暂时不可用。";
    connection.classList.add("offline");
  }
}

refreshView();
setInterval(refreshView, 10000);
