import numpy as np
from PIL import Image
from scipy.io import wavfile


class Signal:
	rate = 96000  # samples/sec

	def __init__(self, samples=np.empty(0, np.float32)):
		self.samples = np.float32(samples)

	def __len__(self):
		return len(self.samples)

	def __iadd__(self, other):
		self.insert(other, 0)
		return self

	def __mul__(self, n):
		return Signal(self.samples.copy() * n)

	def __rmul__(self, n):
		return self * n

	def __truediv__(self, n):
		return Signal(self.samples.copy() / n)

	def copy(self):
		return Signal(self.samples.copy())

	def expand(self, length):
		# length is in samples
		# Might expand the internal sample array, but never shrinks it
		if length > len(self):
			self.samples = np.pad(self.samples, (0, length-len(self)))
		return len(self.samples)

	def insert(self, other, t):
		# t is in seconds
		a = int(t * Signal.rate)
		b = a + len(other)
		self.expand(b)
		self.samples[a:b] += other.samples

	def write(self, filename):
		wavfile.write(filename, Signal.rate, self.samples)

	@staticmethod
	def sine(freq, dur):
		# dur is in seconds
		t = np.linspace(0, dur, int(dur*Signal.rate), dtype=np.float32)
		return Signal(np.sin(2*np.pi*freq * t))


def lerp(t, a, b):
	return (b-a) * t + a

def log_lerp(t, a, b):
	return 10.0 ** lerp(t, np.log10(a), np.log10(b))


print("Building image...")
msg = Signal()
dur = 10  # seconds
low = 24000  # Hz
high = Signal.rate//2 - 1000  # Hz
im = Image.open("sources/bird.png").convert("L")
for x in range(im.width):
	for y in range(im.height):
		freq = lerp(y/im.height, high, low)
		t = lerp(x/im.width, 0, dur)
		px = im.getpixel((x, y))
		amp = log_lerp(px/255, 1e-5, 5e-3) * (1 if y%2 else -1)
		msg.insert(amp * Signal.sine(freq, dur/im.width), t)
		
print("Writing output...")
msg.write("output.wav")
print(f"** Generated {dur} sec of audio between {low} Hz and {high} Hz")