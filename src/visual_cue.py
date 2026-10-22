"""Silent on-screen cues for live captures (the user works in a library: no `say`, no terminal bell).

A small always-on-top Tk window shows a big countdown and the current instruction; the same text is printed
to the terminal. Nothing here produces sound. If Tk cannot open a window (no display), it degrades to
terminal output only.
"""
import sys
import time


class Cue:
    def __init__(self, title="SonarANE", voice=False):
        self.root = None
        self.voice = voice                                      # opt-in only (library rule: silent by default)
        self._say = None
        try:
            import tkinter as tk
            self.root = tk.Tk()
            self.root.title(title)
            self.root.attributes("-topmost", True)
            self.root.geometry("680x300+80+80")
            self.root.configure(bg="#1a1a19")
            self.big = tk.Label(self.root, text="", font=("Helvetica", 72, "bold"), fg="#f0efec", bg="#1a1a19")
            self.big.pack(expand=True)
            self.small = tk.Label(self.root, text="", font=("Helvetica", 22, "bold"), fg="#c3c2b7", bg="#1a1a19",
                                  wraplength=600, justify="center")
            self.small.pack(pady=(0, 24))
            self.pump()
        except Exception as e:                                   # no display / no Tk: terminal only
            print(f"[cue] no window ({e}); terminal cues only", flush=True)
            self.root = None

    def show(self, big, small="", color="#f0efec", bg="#1a1a19"):
        if (big, small) != getattr(self, "_last", None):          # print only when the cue changes
            print(f"===== {big}  {small} =====", flush=True)
            self._last = (big, small)
        if self.root is not None:
            dark = bg == "#1a1a19"                                   # on a coloured background: dark text
            self.root.configure(bg=bg)
            self.big.configure(text=big, fg=color if dark else "#1a1a19", bg=bg)
            self.small.configure(text=small, fg="#c3c2b7" if dark else "#1a1a19", bg=bg)
            self.pump()

    def speak(self, text):
        """Spoken guidance via macOS `say`, non-blocking, only when the Cue was created with voice=True."""
        if self.voice:
            import subprocess
            self._say = subprocess.Popen(["say", "-r", "175", text])

    def pump(self, seconds=0.0):
        """Keep the window responsive for `seconds` (use instead of time.sleep while it is open)."""
        end = time.perf_counter() + seconds
        while True:
            if self.root is not None:
                try:
                    self.root.update()
                except Exception:
                    self.root = None
            if time.perf_counter() >= end:
                break
            time.sleep(0.02)

    def countdown(self, instruction, seconds=3):
        for k in range(seconds, 0, -1):
            self.show(str(k), instruction, "#c98500")
            self.pump(1.0)

    def close(self, final="Done", seconds=1.0):
        self.show(final, "capture finished — you can relax", "#3987e5")
        self.pump(seconds)
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None


def banner(text, countdown=0):
    """One-shot silent cue: window + terminal; optional countdown before returning."""
    c = Cue()
    if countdown:
        c.countdown(text, countdown)
        c.show("GO", text, "#199e70")
        c.pump(0.4)
        c.close("GO", 0.1)
    else:
        c.show("•", text)
        c.pump(1.2)
        c.close(text[:12], 0.1)


if __name__ == "__main__":
    banner(" ".join(sys.argv[1:]) or "SonarANE visual cue test", countdown=3)
