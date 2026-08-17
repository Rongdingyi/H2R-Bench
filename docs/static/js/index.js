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
})();
