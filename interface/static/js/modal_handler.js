/* ===========================================================
   modal_handler.js
   -----------------------------------------------------------
   Persistent modal (response box) that stays until closed.
   Works across Chrome, Safari, and Edge.
   =========================================================== */

function showModal(message, title = "") {
  const modal = document.getElementById("modal");
  const modalText = document.getElementById("modal-text");

  if (!modal || !modalText) {
    console.warn("⚠️ Modal not found in DOM");
    alert(message);
    return;
  }

  modalText.innerHTML = title
    ? `<strong>${title}</strong><br><br>${message}`
    : message;

  modal.style.display = "flex";
  modal.classList.add("visible");
  modal.classList.remove("hidden");
}

function closeModal() {
  const modal = document.getElementById("modal");
  if (modal) {
    modal.classList.remove("visible");
    modal.classList.add("hidden");
    modal.style.display = "none";
  }
}
