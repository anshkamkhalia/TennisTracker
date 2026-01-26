const API_URL = "http://localhost:5000"; // dev
// Use your IP for phone testing

export async function processVideo(uri: string) {
  const formData = new FormData();

  formData.append("video", {
    uri,
    name: "input.mp4",
    type: "video/mp4",
  } as any);

  const res = await fetch(`${API_URL}/process-video`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || "Upload failed");
  }

  return res.json();
}
