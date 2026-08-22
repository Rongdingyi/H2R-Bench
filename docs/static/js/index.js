(() => {
  const toggle = document.querySelector("[data-nav-toggle]");
  const nav = document.querySelector("[data-site-nav]");

  const setMenuOpen = (open) => {
    if (!toggle || !nav) return;
    toggle.setAttribute("aria-expanded", String(open));
    nav.classList.toggle("is-open", open);
  };

  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      setMenuOpen(toggle.getAttribute("aria-expanded") !== "true");
    });

    nav.addEventListener("click", (event) => {
      if (event.target.closest("a")) setMenuOpen(false);
    });

    document.addEventListener("click", (event) => {
      if (!nav.contains(event.target) && !toggle.contains(event.target)) {
        setMenuOpen(false);
      }
    });

    document.addEventListener("keydown", (event) => {
      if (
        event.key === "Escape" &&
        toggle.getAttribute("aria-expanded") === "true"
      ) {
        setMenuOpen(false);
        toggle.focus();
      }
    });

    window.addEventListener("resize", () => {
      if (window.innerWidth > 920) setMenuOpen(false);
    });
  }

  const copyButton = document.querySelector("[data-copy-bibtex]");
  const bibtex = document.querySelector("#bibtex code");

  if (copyButton && bibtex) {
    const label = copyButton.querySelector("[data-copy-label]");

    copyButton.addEventListener("click", async () => {
      const citation = bibtex.textContent.trim();
      try {
        await navigator.clipboard.writeText(citation);
        label.textContent = "Copied";
      } catch {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(bibtex);
        selection.removeAllRanges();
        selection.addRange(range);
        label.textContent = "Selected — press Ctrl/Cmd+C";
      }

      window.setTimeout(() => {
        label.textContent = "Copy BibTeX";
      }, 2200);
    });
  }

  const navLinks = [...document.querySelectorAll('.site-nav a[href^="#"]')];
  const observedSections = navLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  if ("IntersectionObserver" in window && observedSections.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

        if (!visible) return;
        navLinks.forEach((link) => {
          const active = link.getAttribute("href") === `#${visible.target.id}`;
          link.classList.toggle("is-active", active);
          if (active) link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
      },
      {
        rootMargin: "-28% 0px -58% 0px",
        threshold: [0, 0.2, 0.6],
      },
    );

    observedSections.forEach((section) => observer.observe(section));
  }

  /* Tabbed transfer gallery: ARIA roles + roving focus + play only what is visible */
  const tablist = document.querySelector("[data-gallery-tabs]");

  if (tablist) {
    const tabs = Array.from(tablist.querySelectorAll("[data-gallery-tab]"));
    const panels = new Map(
      Array.from(document.querySelectorAll("[data-gallery-panel]")).map(
        (panel) => [panel.dataset.galleryPanel, panel],
      ),
    );

    tablist.setAttribute("role", "tablist");

    tabs.forEach((tab, index) => {
      const key = tab.dataset.galleryTab;
      const panel = panels.get(key);
      if (!panel) return;
      tab.id = tab.id || `gallery-tab-${key}`;
      panel.id = panel.id || `gallery-panel-${key}`;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-controls", panel.id);
      panel.setAttribute("role", "tabpanel");
      panel.setAttribute("aria-labelledby", tab.id);
      panel.setAttribute("tabindex", "0");
      const active = index === 0;
      tab.setAttribute("aria-selected", String(active));
      tab.setAttribute("tabindex", active ? "0" : "-1");
    });

    const select = (key, { focus = false } = {}) => {
      tabs.forEach((tab) => {
        const active = tab.dataset.galleryTab === key;
        tab.classList.toggle("is-active", active);
        tab.setAttribute("aria-selected", String(active));
        tab.setAttribute("tabindex", active ? "0" : "-1");
        if (active && focus) tab.focus();
      });

      panels.forEach((panel, panelKey) => {
        const active = panelKey === key;
        panel.hidden = !active;
        panel.querySelectorAll("video").forEach((video) => {
          if (!active) {
            video.pause();
            return;
          }
          if (video.preload === "none") video.preload = "metadata";
          const play = video.play();
          if (play && typeof play.catch === "function") play.catch(() => {});
        });
      });
    };

    tablist.addEventListener("click", (event) => {
      const tab = event.target.closest("[data-gallery-tab]");
      if (tab) select(tab.dataset.galleryTab);
    });

    tablist.addEventListener("keydown", (event) => {
      const keys = ["ArrowLeft", "ArrowRight", "Home", "End"];
      if (!keys.includes(event.key)) return;
      const current = tabs.findIndex(
        (tab) => tab.getAttribute("aria-selected") === "true",
      );
      let next = current;
      if (event.key === "ArrowLeft") next = (current - 1 + tabs.length) % tabs.length;
      if (event.key === "ArrowRight") next = (current + 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      event.preventDefault();
      select(tabs[next].dataset.galleryTab, { focus: true });
    });
  }

  /* Click any paper figure to open it full size */
  const dialog = document.querySelector("[data-lightbox-dialog]");

  if (dialog && typeof dialog.showModal === "function") {
    const image = dialog.querySelector("[data-lightbox-image]");
    let lastFocus = null;

    const close = () => {
      if (dialog.open) dialog.close();
    };

    document.addEventListener("click", (event) => {
      const trigger = event.target.closest("[data-lightbox]");
      if (!trigger) return;
      event.preventDefault();
      lastFocus = trigger;
      image.src = trigger.dataset.lightbox;
      image.alt = trigger.dataset.lightboxAlt || "";
      dialog.showModal();
      document.body.classList.add("dialog-open");
    });

    dialog.addEventListener("click", (event) => {
      if (event.target === dialog || event.target.closest("[data-lightbox-close]"))
        close();
    });

    dialog.addEventListener("close", () => {
      document.body.classList.remove("dialog-open");
      image.removeAttribute("src");
      if (lastFocus) lastFocus.focus();
    });
  }
})();
