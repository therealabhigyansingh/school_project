// Mark the nav link matching the current page as active.
const path = location.pathname.split("/").pop() || "index.html";
for (const a of document.querySelectorAll(".app-nav a")) {
  if (a.getAttribute("href") === path) a.classList.add("is-active");
}
