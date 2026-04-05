import { useLocation, useNavigate } from "react-router-dom";
import { ArrowLeft, Video, Home } from "lucide-react";

export default function Result() {
  const location = useLocation();
  const navigate = useNavigate();
  const url = (location.state as any)?.url;

  if (!url) {
    return (
      <div className="px-10 py-20 text-center">
        <p className="text-6xl mb-4">🎾</p>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">No video found</h2>
        <p className="text-gray-500 mb-6">Go back and upload a video first.</p>
        <button
          onClick={() => navigate("/record")}
          className="bg-[#1A3263] text-white border-none rounded-xl px-6 py-3 font-semibold cursor-pointer hover:bg-[#15295a] transition"
        >
          Go to Record
        </button>
      </div>
    );
  }

  return (
    <div className="px-10 py-8 max-w-6xl mx-auto">
      <div className="flex items-center gap-4 mb-8">
        <button
          onClick={() => navigate("/record")}
          className="flex items-center gap-2 bg-white border border-gray-200 rounded-lg px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 transition cursor-pointer"
        >
          <ArrowLeft size={15} /> Back to Record
        </button>
        <div>
          <h1 className="text-2xl font-extrabold text-gray-900">Analysis Results</h1>
          <p className="text-sm text-gray-500 mt-0.5">Your AI-powered session breakdown</p>
        </div>
      </div>

      <div>
        <div>
          <div className="bg-black rounded-2xl overflow-hidden aspect-video mb-4">
            <video src={url} controls autoPlay className="w-full h-full object-contain" />
          </div>
          <div className="flex gap-3">
            <button
              onClick={() => navigate("/home")}
              className="flex-1 flex items-center justify-center gap-2 bg-white border border-gray-200 rounded-xl py-3 text-gray-700 font-semibold text-sm hover:bg-gray-50 transition cursor-pointer"
            >
              <Home size={15} /> Go Home
            </button>
            <button
              onClick={() => navigate("/record")}
              className="flex-1 flex items-center justify-center gap-2 bg-[#1A3263] border-none rounded-xl py-3 text-white font-semibold text-sm hover:bg-[#15295a] transition cursor-pointer"
            >
              <Video size={15} /> New Recording
            </button>
          </div>
        </div>

        
      </div>
    </div>
  );
}