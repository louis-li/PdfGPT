function UploadDocument() {
  const modal = document.getElementById("modal");
  modal.classList.toggle("hidden");


  const form = document.getElementById("update-form");
  if (form) {
    form.addEventListener("submit", function(event) {
      event.preventDefault();

      var xhr = new XMLHttpRequest();
      xhr.open("POST", form.action);
      xhr.send(new FormData(form));
      modal.classList.toggle("hidden");
    });
  }
}
function cancelUpload() {
  const modal = document.getElementById("modal");

  if (form) {
    form.addEventListener("submit", function(event) {
      event.preventDefault();
    });
  }

  modal.classList.toggle("hidden");

}

modal.classList.toggle("hidden");

