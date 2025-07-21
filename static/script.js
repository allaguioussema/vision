document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const fileInput = document.getElementById("file-input");
  const uploadArea = document.getElementById("upload-area");
  const detectBtn = document.getElementById("detect-btn");
  const resetBtn = document.getElementById("reset-btn");
  const detectionTypeSelect = document.getElementById("detection-type-select");
  const preview = document.getElementById("preview");
  const jsonResults = document.getElementById("json-results");
  const jsonContent = document.getElementById("json-content");
  const statusDiv = document.getElementById("status");

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

    preview.style.display = "none";
    jsonResults.style.display = "none";
    preview.src = "";

    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.style.display = "block";
  }

  // ========= Detect Button Logic =========
  detectBtn.addEventListener("click", async () => {
    if (!fileInput.files.length) {
      return showStatus("Please select a file first", "error");
    }

    // Hide upload area on detect
    uploadArea.style.display = "none";

    const file = fileInput.files[0];
    const detectionType = detectionTypeSelect.value;

    detectBtn.innerHTML =
      '<i class="fas fa-spinner fa-spin"></i> Processing...';
    detectBtn.disabled = true;
    showStatus("Processing...", "loading");

    try {
      const formData = new FormData();
      formData.append("file", file);

      let endpoint;
      let isJsonResponse = false;

      // Determine endpoint based on detection type
      switch (detectionType) {
        case "ingredient":
          endpoint = "/detect/ingredient";
          break;
        case "nutrition":
          endpoint = "/detect/nutrition";
          break;
        case "both":
          endpoint = "/detect/both";
          isJsonResponse = true;
          break;
        default:
          throw new Error("Invalid detection type");
      }

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      if (isJsonResponse) {
        // Handle JSON response for "both" detection
        const data = await response.json();
        
        // Hide image preview and show JSON results
        preview.style.display = "none";
        jsonResults.style.display = "block";
        jsonContent.textContent = JSON.stringify(data, null, 2);
        
        showStatus("Detection completed! Check results below.", "success");
      } else {
        // Handle image response for single detection
        const blob = await response.blob();
        const detectedURL = URL.createObjectURL(blob);

        preview.src = detectedURL;
        preview.style.display = "block";
        jsonResults.style.display = "none";

        showStatus("Detection completed!", "success");
      }
    } catch (error) {
      console.error(error);
      showStatus(error.message, "error");
    } finally {
      detectBtn.innerHTML = '<i class="fas fa-search"></i> Detect';
      detectBtn.disabled = false;
    }
  });

  // ========= Reset Button Logic =========
  resetBtn.addEventListener("click", () => {
    fileInput.value = "";
    preview.src = "";
    preview.style.display = "none";
    jsonResults.style.display = "none";
    statusDiv.style.display = "none";
    uploadArea.style.display = "block"; // Show upload area again
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
  });
});
