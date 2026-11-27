"""Consent-first file sharing window for nearby GestureFlow AI devices."""

from __future__ import annotations

import threading
import queue
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from gestureflow_ai.device_sharing import DeviceShareReceiver, send_file, send_gesture_state


class DeviceShareWindow(tk.Toplevel):
    """Prototype UI for manually paired, same-network file handoff."""

    def __init__(self, parent, *, on_arm=None):
        super().__init__(parent)
        self.parent = parent
        self._on_arm = on_arm
        self.title("GestureFlow AI · Device Share")
        self.geometry("580x650")
        self.minsize(520, 600)
        self.configure(bg="#091627")
        self.transient(parent)
        self.receiver = DeviceShareReceiver()
        self.selected_file = tk.StringVar(value="No file selected")
        self.host = tk.StringVar()
        self.port = tk.StringVar(value="8765")
        self.code = tk.StringVar()
        self.status = tk.StringVar(value="Choose a file or start this device as a receiver.")
        self._poll_id = None
        self._send_results: queue.Queue[tuple[bool, str]] = queue.Queue()
        self._gesture_jobs: queue.Queue[tuple] = queue.Queue()
        self._gesture_armed = False
        self._gesture_track_id = None
        self._gesture_waiting_for_open = False
        self._gesture_holding = False
        self._gesture_missing_since = None
        self._gesture_last_source_seq = None
        self._gesture_worker = threading.Thread(target=self._gesture_sender, daemon=True)
        self._gesture_worker.start()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._poll_id = self.after(250, self._poll_incoming)

    def _build(self):
        style = ttk.Style(self)
        style.configure("Share.TFrame", background="#213a55")
        style.configure("ShareTitle.TLabel", background="#213a55", foreground="#edf5ff",
                        font=("Segoe UI", 14, "bold"))
        style.configure("ShareText.TLabel", background="#213a55", foreground="#b8c9dc",
                        font=("Segoe UI", 9))
        style.configure("ShareAccent.TLabel", background="#213a55", foreground="#83e8ff",
                        font=("Segoe UI", 11, "bold"))
        outer = ttk.Frame(self, style="Share.TFrame", padding=20)
        outer.pack(fill="both", expand=True, padx=16, pady=16)
        outer.columnconfigure(0, weight=1)
        ttk.Label(outer, text="Nearby device handoff", style="ShareTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            outer,
            text="Choose a file, pair with the receiver on the same Wi-Fi, then show an open hand → close a fist to hold → open your hand to send.",
            style="ShareText.TLabel", wraplength=440, justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(5, 14))

        file_row = ttk.Frame(outer, style="Share.TFrame")
        file_row.grid(row=2, column=0, sticky="ew")
        file_row.columnconfigure(0, weight=1)
        ttk.Label(file_row, textvariable=self.selected_file, style="ShareText.TLabel",
                  wraplength=320).grid(row=0, column=0, sticky="w")
        ttk.Button(file_row, text="Choose file", command=self._choose_file).grid(
            row=0, column=1, padx=(8, 0)
        )

        receiver = ttk.LabelFrame(outer, text="Receive on this device", padding=12)
        receiver.grid(row=3, column=0, sticky="ew", pady=(14, 8))
        receiver.columnconfigure(0, weight=1)
        self.receiver_details = ttk.Label(
            receiver,
            text=("Receiver is off. Accepted files will be saved here:\n"
                  f"{self.receiver.receive_directory}"),
            style="ShareText.TLabel", wraplength=400,
        )
        self.receiver_details.grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.receiver_button = ttk.Button(receiver, text="Start receiving", command=self._toggle_receiver)
        self.receiver_button.grid(row=1, column=0, sticky="w")
        self.open_folder_button = ttk.Button(
            receiver, text="Open received files", command=self._open_received_folder
        )
        self.open_folder_button.grid(row=1, column=1, sticky="e", padx=(8, 0))

        sender = ttk.LabelFrame(outer, text="Send to another device", padding=12)
        sender.grid(row=4, column=0, sticky="ew", pady=8)
        sender.columnconfigure(1, weight=1)
        for row, label, variable in (
            (0, "Device IP", self.host),
            (1, "Port", self.port),
            (2, "Pairing code", self.code),
        ):
            ttk.Label(sender, text=label, style="ShareText.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=3
            )
            ttk.Entry(sender, textvariable=variable, show="*" if label == "Pairing code" else "").grid(
                row=row, column=1, sticky="ew", pady=3
            )
        self.send_button = ttk.Button(sender, text="Send selected file", command=self._send)
        self.send_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(9, 0))

        self.gesture_button = ttk.Button(
            sender, text="Arm open → fist → open hand transfer",
            command=self._toggle_gesture_handoff,
        )
        self.gesture_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(7, 0))

        self.remote_status = ttk.Label(
            receiver, text="No remote handoff in progress.", style="ShareAccent.TLabel",
            wraplength=400,
        )
        self.remote_status.grid(row=2, column=0, sticky="w", pady=(10, 0))

        ttk.Label(outer, textvariable=self.status, style="ShareAccent.TLabel", wraplength=440,
                  justify="left").grid(row=5, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(
            outer,
            text="Prototype uses HTTP on your local network. Use it only on a trusted Wi-Fi network; every received file needs approval.",
            style="ShareText.TLabel", wraplength=440, justify="left",
        ).grid(row=6, column=0, sticky="ew", pady=(10, 0))

    def _choose_file(self):
        path = filedialog.askopenfilename(parent=self, title="Choose a file to share")
        if path:
            self.selected_file.set(path)
            self.status.set("File selected. Enter the receiving device details to send.")

    def _toggle_gesture_handoff(self):
        if self._gesture_armed:
            self._cancel_gesture_handoff("Gesture handoff cancelled.")
            return
        if self.selected_file.get() == "No file selected":
            messagebox.showinfo("Choose a file", "Choose a file before arming hand transfer.", parent=self)
            return
        try:
            port = int(self.port.get().strip())
        except ValueError:
            messagebox.showerror("Check the port", "Enter a valid receiver port.", parent=self)
            return
        if not self.host.get().strip() or not self.code.get().strip():
            messagebox.showinfo("Pair with a device", "Enter the receiver IP and pairing code.", parent=self)
            return
        if self._on_arm is not None and not self._on_arm():
            messagebox.showinfo(
                "Start camera detection",
                "Turn on the camera and start hand detection before arming transfer.",
                parent=self,
            )
            return
        self._gesture_armed = True
        self._gesture_track_id = None
        self._gesture_waiting_for_open = True
        self._gesture_holding = False
        self._gesture_missing_since = None
        self._gesture_last_source_seq = None
        self.gesture_button.configure(text="Cancel hand transfer")
        self.status.set("Show one open hand, close it into a fist to hold the selected file.")

    def process_hand_gestures(self, candidates, tracks, source_seq=None):
        """Consume stable gestures from one sticky hand track while explicitly armed."""
        if not self._gesture_armed:
            return
        # UI refreshes can outpace camera inference. Count each detector result once.
        if source_seq is not None:
            if source_seq == self._gesture_last_source_seq:
                return
            self._gesture_last_source_seq = source_seq
        path = self.selected_file.get()
        pairs = list(zip(candidates or (), tracks or ()))
        selected = None
        if self._gesture_track_id is not None:
            selected = next(
                ((candidate, track) for candidate, track in pairs
                 if getattr(track, "track_id", None) == self._gesture_track_id),
                None,
            )
        elif pairs:
            # Bind to a visible hand while its classifier warms up, then wait for
            # a stable open hand before accepting the fist as a hold command.
            open_hands = [
                pair for pair in pairs
                if str(getattr(pair[1], "stable_gesture", "UNKNOWN")).upper()
                .replace("_", " ") == "OPEN HAND"
            ]
            selected = max(
                open_hands or pairs,
                key=lambda pair: float(getattr(pair[0], "quality", 0.0)),
            )
            self._gesture_track_id = getattr(selected[1], "track_id", None)
        if selected is None:
            if self._gesture_missing_since is None:
                self._gesture_missing_since = time.monotonic()
            elif time.monotonic() - self._gesture_missing_since >= 1.5:
                if self._gesture_holding:
                    self._gesture_jobs.put(("state", self.host.get(), int(self.port.get()),
                                            self.code.get(), "cancelled", path))
                self._reset_gesture_state("Hand lost. Show an open hand to arm again.")
            return
        self._gesture_missing_since = None
        _candidate, track = selected
        gesture = str(getattr(track, "stable_gesture", "UNKNOWN") or "UNKNOWN")
        gesture = gesture.upper().replace("_", " ")
        if self._gesture_waiting_for_open and gesture == "OPEN HAND":
            self._gesture_waiting_for_open = False
            self.status.set("Hand selected. Close it into a fist to hold the file.")
        elif not self._gesture_waiting_for_open and not self._gesture_holding and gesture == "CLOSED FIST":
            self._gesture_holding = True
            self._gesture_jobs.put(("state", self.host.get(), int(self.port.get()),
                                    self.code.get(), "holding", path))
            self.status.set("File held. The other device shows it; open your hand to send.")
        elif self._gesture_holding and gesture == "OPEN HAND":
            self._gesture_holding = False
            self._gesture_armed = False
            self.gesture_button.configure(text="Arm open → fist → open hand transfer")
            self._gesture_jobs.put(("state", self.host.get(), int(self.port.get()),
                                    self.code.get(), "released", path))
            self._gesture_jobs.put(("transfer", self.host.get(), int(self.port.get()),
                                    self.code.get(), path))
            self.status.set("Hand opened. Sending file; approve it on the receiving device.")

    def _gesture_sender(self):
        while True:
            job = self._gesture_jobs.get()
            if job[0] == "stop":
                return
            try:
                if job[0] == "state":
                    _, host, port, code, state, path = job
                    send_gesture_state(host, port, code, state, path)
                else:
                    _, host, port, code, path = job
                    result = send_file(host, port, code, path)
                    self._send_results.put((True, result))
            except Exception as exc:
                self._send_results.put((False, str(exc)))

    def _reset_gesture_state(self, message):
        self._gesture_armed = False
        self._gesture_track_id = None
        self._gesture_waiting_for_open = False
        self._gesture_holding = False
        self._gesture_missing_since = None
        self._gesture_last_source_seq = None
        self.gesture_button.configure(text="Arm open → fist → open hand transfer")
        self.status.set(message)

    def _cancel_gesture_handoff(self, message):
        if self._gesture_holding:
            try:
                self._gesture_jobs.put(("state", self.host.get(), int(self.port.get()),
                                        self.code.get(), "cancelled", self.selected_file.get()))
            except ValueError:
                pass
        self._reset_gesture_state(message)

    def _toggle_receiver(self):
        if self.receiver.running:
            self.receiver.stop()
            self.receiver_button.configure(text="Start receiving")
            self.receiver_details.configure(
                text=f"Receiver is off. Received files are in:\n{self.receiver.receive_directory}"
            )
            self.status.set("Receiver stopped.")
            return
        try:
            address, port, code = self.receiver.start()
        except OSError as exc:
            messagebox.showerror("Could not start receiver", str(exc), parent=self)
            return
        self.receiver_button.configure(text="Stop receiving")
        self.receiver_details.configure(
            text=(f"Receiver ready · IP {address} · Port {port}\n"
                  f"Temporary pairing code: {code}\n"
                  f"Accepted files save to: {self.receiver.receive_directory}")
        )
        self.status.set("Share these details with the sender on the same Wi-Fi network.")

    def _open_received_folder(self):
        try:
            self.receiver.receive_directory.mkdir(parents=True, exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(str(self.receiver.receive_directory))
            else:
                messagebox.showinfo(
                    "Received files folder",
                    str(self.receiver.receive_directory),
                    parent=self,
                )
        except OSError as exc:
            messagebox.showerror("Could not open folder", str(exc), parent=self)

    def _send(self):
        path = self.selected_file.get()
        if path == "No file selected":
            messagebox.showinfo("Choose a file", "Choose the file you want to share first.", parent=self)
            return
        try:
            port = int(self.port.get().strip())
        except ValueError:
            messagebox.showerror("Check the port", "Enter a valid port number.", parent=self)
            return
        if not self.host.get().strip() or not self.code.get().strip():
            messagebox.showinfo("Pair with a device", "Enter the receiving device IP and pairing code.", parent=self)
            return
        self.send_button.configure(state="disabled")
        self.status.set("Sending file… approve it on the receiving device.")

        def work():
            try:
                result = send_file(self.host.get(), port, self.code.get(), path)
                outcome = (True, result)
            except Exception as exc:
                outcome = (False, str(exc))
            self._send_results.put(outcome)

        threading.Thread(target=work, daemon=True).start()

    def _send_finished(self, success: bool, message: str):
        self.send_button.configure(state="normal")
        self.status.set(message if success else f"Transfer failed: {message}")

    def _poll_incoming(self):
        if not self.winfo_exists():
            return
        try:
            while True:
                self._send_finished(*self._send_results.get_nowait())
        except queue.Empty:
            pass
        if self.receiver.running:
            remote = self.receiver.remote_gesture_status()
            if remote:
                labels = {
                    "holding": f"{remote['filename']} is being held by {remote['sender']}.",
                    "released": f"{remote['filename']} released · transfer starting…",
                    "cancelled": "Remote handoff cancelled.",
                }
                self.remote_status.configure(text=labels.get(remote["state"], "Remote handoff active."))
            else:
                self.remote_status.configure(text="No remote handoff in progress.")
            incoming = self.receiver.next_request()
            if incoming is not None:
                size_mb = incoming.size / (1024 * 1024)
                approved = messagebox.askyesno(
                    "Incoming file",
                    (f"Accept {incoming.filename} ({size_mb:.2f} MB) from {incoming.sender}?\n\n"
                     f"It will be saved to:\n{self.receiver.receive_directory}"),
                    parent=self,
                )
                self.receiver.decide(incoming.request_id, approved)
                if not approved:
                    self.status.set(f"Declined {incoming.filename}.")
                elif incoming.accepted and incoming.saved_path:
                    self.status.set(f"Saved on this laptop: {incoming.saved_path}")
                    self.remote_status.configure(text=f"Saved here · {incoming.saved_path.name}")
                else:
                    reason = incoming.save_error or "The receiver could not finish saving the file."
                    self.status.set(f"Save failed for {incoming.filename}: {reason}")
        self._poll_id = self.after(250, self._poll_incoming)

    def _close(self):
        if self._poll_id:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
        self.receiver.stop()
        self._cancel_gesture_handoff("Device share window closed.")
        self._gesture_jobs.put(("stop",))
        self.destroy()
