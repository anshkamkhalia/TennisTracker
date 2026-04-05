import { useUser, useClerk } from "@clerk/clerk-react";
import { useNavigate } from "react-router-dom";
import { Camera, ChevronRight, User, Settings, Shield, LogOut, Award, Mail } from "lucide-react";

export default function Profile() {
  const { user } = useUser();
  const { signOut } = useClerk();
  const navigate = useNavigate();

  const displayName = user?.fullName || user?.emailAddresses?.[0]?.emailAddress || "Player";
  const email = user?.emailAddresses?.[0]?.emailAddress || "";
  const initials = user?.firstName && user?.lastName
    ? `${user.firstName[0]}${user.lastName[0]}`
    : displayName[0]?.toUpperCase() ?? "P";

  async function handleLogout() {
    await signOut();
    navigate("/");
  }

  return (
    <div className="px-10 py-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-extrabold text-gray-900 mb-8">My Profile</h1>

      <div className="grid gap-7" style={{ gridTemplateColumns: "1fr 2fr" }}>
        {/* Left — avatar + stats */}
        <div className="flex flex-col gap-5">
          {/* Avatar card */}
          <div className="bg-white rounded-2xl border border-gray-200 p-7 text-center">
            <div className="relative inline-block mb-4">
              {user?.imageUrl ? (
                <img src={user.imageUrl} alt="avatar" className="w-24 h-24 rounded-full object-cover" />
              ) : (
                <div className="w-24 h-24 rounded-full bg-[#1A3263] flex items-center justify-center text-3xl text-white font-bold">
                  {initials}
                </div>
              )}
              <button className="absolute bottom-0 right-0 bg-[#1A3263] border-4 border-white rounded-full p-1.5 cursor-pointer flex items-center justify-center">
                <Camera size={13} color="#fff" />
              </button>
            </div>
            <h2 className="text-lg font-bold text-gray-900">{displayName}</h2>
            <p className="text-gray-500 text-sm mt-1 flex items-center gap-1 justify-center">
              <Mail size={12} /> {email}
            </p>
            <div className="inline-flex items-center gap-1.5 bg-blue-100 rounded-full px-3 py-1 mt-3">
              <Award size={13} className="text-[#1A3263]" />
              <span className="text-xs font-bold text-[#1A3263]">Intermediate Player</span>
            </div>
          </div>


        </div>

        {/* Right — settings */}
        <div className="flex flex-col gap-5">
          {/* Account info */}
          <div className="bg-white rounded-2xl border border-gray-200 p-7">
            <h3 className="text-base font-bold text-gray-900 mb-5">Account Details</h3>
            <div className="flex flex-col gap-4">
              {[
                { label: "Full name", value: displayName },
                { label: "Email address", value: email },
                { label: "Plan", value: "Free" },
                { label: "Member since", value: user?.createdAt ? new Date(user.createdAt).toLocaleDateString("en-US", { month: "long", year: "numeric" }) : "—" },
              ].map(f => (
                <div key={f.label} className="flex justify-between pb-4 border-b border-gray-50 last:border-b-0 last:pb-0">
                  <span className="text-gray-500 text-sm">{f.label}</span>
                  <span className="text-gray-900 text-sm font-semibold">{f.value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Menu */}
          <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden">
            <h3 className="text-base font-bold text-gray-900 px-6 pt-5 pb-4">Settings</h3>
            {[
              { icon: <User size={17} className="text-[#1A3263]" />, bg: "bg-blue-100", label: "Edit Profile", desc: "Update your name and avatar" },
              { icon: <Settings size={17} className="text-amber-600" />, bg: "bg-amber-100", label: "Preferences", desc: "Notifications and app settings" },
              { icon: <Shield size={17} className="text-purple-600" />, bg: "bg-purple-100", label: "Privacy & Security", desc: "Password, sessions, data" },
            ].map((item, i, arr) => (
              <button key={item.label} className={`flex items-center gap-4 w-full px-6 py-3.5 border-none bg-white cursor-pointer text-left hover:bg-gray-50 transition ${i < arr.length - 1 ? "border-b border-gray-50" : ""}`}>
                <div className={`w-10 h-10 rounded-xl ${item.bg} flex items-center justify-center shrink-0`}>
                  {item.icon}
                </div>
                <div className="flex-1">
                  <p className="font-semibold text-gray-900 text-sm">{item.label}</p>
                  <p className="text-gray-400 text-xs mt-0.5">{item.desc}</p>
                </div>
                <ChevronRight size={16} className="text-gray-300" />
              </button>
            ))}

            <div className="border-t border-gray-100 mx-6" />

            <button
              onClick={handleLogout}
              className="flex items-center gap-4 w-full px-6 py-3.5 border-none bg-white cursor-pointer text-left hover:bg-red-50 transition"
            >
              <div className="w-10 h-10 rounded-xl bg-red-50 flex items-center justify-center shrink-0">
                <LogOut size={17} className="text-red-500" />
              </div>
              <div className="flex-1">
                <p className="font-semibold text-red-500 text-sm">Sign Out</p>
                <p className="text-gray-400 text-xs mt-0.5">You'll be returned to the homepage</p>
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}