document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const fileInput = document.getElementById("file-input");
  const uploadArea = document.getElementById("upload-area");
  const detectBtn = document.getElementById("detect-btn");
  const resetBtn = document.getElementById("reset-btn");
  const modelSelect = document.getElementById("model-select");
  const preview = document.getElementById("preview");
  const videoPreview = document.getElementById("video-preview");
  const statusDiv = document.getElementById("status");

  let currentStreamUrl = "";

  // ========= Upload Area Interactions =========
  uploadArea.addEventListener("click", () => fileInput.click());

  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("active");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("active");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("active");

    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileChange();
    }
  });

  fileInput.addEventListener("change", handleFileChange);

  function handleFileChange() {
    const file = fileInput.files[0];
    document.getElementById("file-info").textContent = `${
      file.name
    } (${(file.size / 1024).toFixed(2)} KB)`;

    if (!file) return;

    const isVideo = file.type.includes("video");

    preview.style.display = "none";
    videoPreview.style.display = "none";
    preview.src = "";
    videoPreview.src = "";

    const url = URL.createObjectURL(file);

    if (isVideo) {
      videoPreview.src = url;
      videoPreview.style.display = "block";
      videoPreview.load();
    } else {
      preview.src = url;
      preview.style.display = "block";
    }
  }

  // ========= Detect Button Logic =========
  detectBtn.addEventListener("click", async () => {
    if (!fileInput.files.length) {
      return showStatus("Please select a file first", "error");
    }

    // Hide upload area on detect
    uploadArea.style.display = "none";

    const file = fileInput.files[0];
    const isVideo = file.type.includes("video");
    const model = modelSelect.value;

    detectBtn.innerHTML =
      '<i class="fas fa-spinner fa-spin"></i> Processing...';
    detectBtn.disabled = true;
    showStatus("Processing...", "loading");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`/detect?model=${model}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      if (isVideo) {
        const data = await response.json();
        if (!data.stream_url) throw new Error("Stream URL not found");

        currentStreamUrl = data.stream_url;
        preview.src = currentStreamUrl;
        preview.style.display = "block";

        videoPreview.pause();
        videoPreview.src = "";
        videoPreview.style.display = "none";

        showStatus("Streaming started...", "success");
      } else {
        const blob = await response.blob();
        const detectedURL = URL.createObjectURL(blob);

        preview.src = detectedURL;
        preview.style.display = "block";
        videoPreview.style.display = "none";

        showStatus("Detection completed!", "success");
      }
    } catch (error) {
      console.error(error);
      showStatus(error.message, "error");
    } finally {
      detectBtn.innerHTML = '<i class="fas fa-search"></i> Detect Objects';
      detectBtn.disabled = false;
    }
  });

  // ========= Reset Button Logic =========
  resetBtn.addEventListener("click", () => {
    fileInput.value = "";
    preview.src = "";
    videoPreview.src = "";
    preview.style.display = "none";
    videoPreview.style.display = "none";
    statusDiv.style.display = "none";
    uploadArea.style.display = "block"; // Show upload area again
    currentStreamUrl = "";
    document.getElementById("file-info").textContent = "";
  });

  // ========= Status Display Helper =========
  function showStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    statusDiv.style.display = "block";
  }

  // ========= Cleanup on Unload =========
  window.addEventListener("beforeunload", () => {
    if (preview.src) URL.revokeObjectURL(preview.src);
    if (videoPreview.src) URL.revokeObjectURL(videoPreview.src);
  });
});
