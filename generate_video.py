import os
import argparse
import cv2
import mido
import numpy as np

# Compatible keyboard list
KEYBOARDS = {}
KEYBOARDS["88"] = {"start_key": 21}
KEYBOARDS["76"] = {"start_key": 28}
KEYBOARDS["61"] = {"start_key": 36}
KEYBOARDS["49"] = {"start_key": 36}
KEYBOARDS["37-C"] = {"start_key": 48}
KEYBOARDS["37-F"] = {"start_key": 41}
KEYBOARDS["36-C"] = {"start_key": 48}
KEYBOARDS["36-F"] = {"start_key": 41}
KEYBOARDS["32-C"] = {"start_key": 48}
KEYBOARDS["32-F"] = {"start_key": 41}


# Default parameters values
DEFAULT_KEYBOARD = "88"
DEFAULT_FPS = 25
DEFAULT_MONITOR_DURATION = 3.0 # seconds
DEFAULT_START_OFFSET = 3.0 # seconds
DEFAULT_END_OFFSET = 1.0 # seconds
VIDEO_HEIGHT_RATIO = 0.5625 # = 1080 / 1920


def get_args():
    parser = argparse.ArgumentParser(description="Script to generate a video showing piano keys to play from a midi file")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path of the input midi file [.mid]")
    parser.add_argument('--keyboard', type=str, required=False, default=DEFAULT_KEYBOARD,
                        help="Keyboard model to use, compatible keyboards are %s" % list(KEYBOARDS.keys()))
    parser.add_argument('--fps', type=int, required=False, default=DEFAULT_FPS,
                        help="Frame per second of the output video generated")
    parser.add_argument('--monitor_duration', type=int, required=False, default=DEFAULT_MONITOR_DURATION,
                        help="Length of the monitor in seconds (Number of seconds represented). A small value will increase the speed of the 'rolling out' note but will also decrease the time the note are seen in advance")
    parser.add_argument('--start_offset', type=int, required=False, default=DEFAULT_START_OFFSET,
                        help="Time offset to add at the beginning of the video")
    parser.add_argument('--end_offset', type=int, required=False, default=DEFAULT_END_OFFSET,
                        help="Time offset to add at the end of the video")
    parser.add_argument('--height_ratio', type=int, required=False, default=VIDEO_HEIGHT_RATIO,
                        help="Currently the height of the monitor showing the notes to play is set so that the height / with ratio is 16/9")
    parser.add_argument('-o', '--output', type=str, required=False,
                        help="Path of the output video file (.mp4)")
                        
    args = parser.parse_args()

    path_file_in, input_ext = os.path.splitext(args.input)

    if input_ext.upper() != ".MID":
        raise ValueError("Invalid input file format '%s', The input file must be a .mid file" % input_ext)
    if args.output is None:
        args.output = path_file_in + ".mp4"
    elif os.path.isdir(args.output):
        args.output = os.path.join(args.output, os.path.basename(path_file_in) + ".mp4")
    else:
        output_ext = os.path.splitext(args.output)[1]
        if(output_ext.upper() == ""):
            args.output += ".mp4"
        elif(output_ext.upper() != ".MP4"):
            raise ValueError("Invalid output file format '%s', The output file must be a .mp4 video file" % output_ext)
    return args


class MidiLoader:
    def __init__(self):
        self.PPQ = None
        self.tick_durations = None

    def calcuateTickDuration(self, tempo):
        BPM = 6e7 / tempo
        tick_duration = (60 / BPM) / self.PPQ
        return tick_duration

    def getTempoChanges(self, midi):
        self.tick_durations = []
        for track in midi.tracks:
            nb_ticks = 0
            for message in track:
                nb_ticks += message.time
                if hasattr(message, "tempo"):
                    duration = self.calcuateTickDuration(message.tempo)
                    self.tick_durations.append({"index": nb_ticks, "value": duration})
        self.tick_durations.sort(key=lambda x:x["index"])
        if len(self.tick_durations) == 0:
            print("Tempo has not been found, setting it to default value (BPM = 120)")
            duration = self.calcuateTickDuration(500000)
            self.tick_durations.append({"index": 0, "value": duration})
        if self.tick_durations[0]["index"] > 0:
            print("The first tempo message occured after time=0...bad ! Setting it to 0 manually")
            self.tick_durations[0]["index"] = 0

    def getTickDuration(self, tick):
        durations = [duration["value"] for duration in self.tick_durations if duration["index"] <= tick]
        return durations[-1]

    def isNoteOn(self, message):
        if(message.type.upper() == "NOTE_ON"):
            if message.velocity != 0:
                return True
        return False

    def isNoteOff(self, message):
        if(message.type.upper() == "NOTE_OFF"):
            return True
        elif(message.type.upper() == "NOTE_ON"):
            if message.velocity == 0:
                return True
        return False

    def extract(self, path):
        # Opening the midi file
        midi = mido.MidiFile(path)
        # Get the Pulse Per Quarter Note info
        self.PPQ = midi.ticks_per_beat
        # Get indexes where there are changes of tempo
        self.getTempoChanges(midi)
        tracks = []
        for track in midi.tracks:
            notes = []
            nb_ticks = 0
            t = 0
            notes_on_time = {}
            for message in track:
                nb_ticks += message.time
                tick_duration = self.getTickDuration(nb_ticks)
                t += tick_duration * message.time
                if self.isNoteOn(message):
                    if message.note not in notes_on_time:
                        notes_on_time[message.note] = t
                elif self.isNoteOff(message):
                    if message.note in notes_on_time:
                        duration = t - notes_on_time[message.note]
                        notes.append({"starting": notes_on_time[message.note], "duration": duration, "key": message.note})
                        notes_on_time.pop(message.note)
                    else:
                        print("`Note off` message for a note not existing...hum..?")
            tracks.append(notes)
        return tracks



class PianoModel:
    def __init__(self, keyboard="88"):
        if keyboard not in KEYBOARDS:
            raise ValueError("Keyboard `%s` does not exist." % keyboard)
        self.first_key = KEYBOARDS[keyboard]["start_key"]
        self.nb_keys = int(keyboard[0:2])
        self.last_key = self.first_key + self.nb_keys - 1
        ### Piano design (Begin) ###
        # White keys
        self.white_keys_height = 200
        self.white_keys_width = 30
        # Black keys
        self.black_keys_height = 133
        self.black_keys_width = 20
        self.black_keys_uncentering = 3
        # Red band
        self.red_band_height = 3
        self.red_band_color = np.array([168, 0, 0], dtype=np.uint8)
        # Global parameters
        self.keys_margin = 2 # If odd/even number then self.black_keys_width should be odd/even number too
        self.background_color = np.array([139, 139, 139], dtype=np.uint8)
        ### Piano design (End) ###
        # Image parameters
        self.nb_white_keys = self.getNumberOfWhiteKeys()
        self.img_height = self.white_keys_height + self.red_band_height
        self.img_width = self.white_keys_width * self.nb_white_keys + self.keys_margin * (self.nb_white_keys - 1)
        # Draw the Immutable keyboard model
        self.drawKeyboard()

    def keysIterator(self):
        for key in range(self.first_key, self.last_key + 1):
            yield key

    def isWhiteKey(self, key):
        return (key % 12) in [0, 2, 4, 5, 7, 9, 11]

    def getNumberOfWhiteKeys(self, key_stop=None):
        nb_white_keys = 0
        for key in self.keysIterator():
            if key_stop == key:
                break
            if self.isWhiteKey(key):
                nb_white_keys += 1
        return nb_white_keys

    def getXStart(self, key):
        nb_white = self.getNumberOfWhiteKeys(key_stop=key)
        white_start = nb_white * (self.white_keys_width + self.keys_margin)
        # White key
        if self.isWhiteKey(key):
            return white_start
        # Black key
        x_black_middle = white_start - self.keys_margin
        x_black_middle += int(self.keys_margin / 2)
        x_black_middle -= int(self.black_keys_width / 2)
        if (key % 12) in [1, 6]:
            # DO# or FA# (C# Or F#)
            return x_black_middle - self.black_keys_uncentering
        elif (key % 12) == 8:
            # SOL# (G#)
            return x_black_middle
        elif (key % 12) in [3, 10]:
            # MIb or SIb (Eb Or Bb)
            return x_black_middle + self.black_keys_uncentering
        else:
            raise ValueError("This error should never happen")

    def getCoordinates(self, key):
        xmin = self.getXStart(key)
        xmax = xmin + self.white_keys_width if self.isWhiteKey(key) else xmin + self.black_keys_width
        return (xmin, xmax)

    def getShape(self):
        return (self.img_height, self.img_width)

    def addColor(self, img, color, key_weight=0.5, color_weight=0.5):
        mask = np.ones(img.shape, dtype=np.uint8) * np.array(color, dtype=np.uint8)
        return cv2.addWeighted(img, key_weight, mask, color_weight, 0.0)

    def getWhiteKeyMask(self, key, pressed=False):
        key_img_mask = np.zeros((self.white_keys_height, self.white_keys_width) , dtype=bool)
        if ((key % 12) in [0, 5]) and (key != self.last_key):
            # DO or FA (C Or F)
            x1 = self.white_keys_width - int(self.black_keys_width / 2) - self.keys_margin - self.black_keys_uncentering + int(self.keys_margin / 2)
            x2 = self.white_keys_width
            key_img_mask[0 : self.black_keys_height + self.keys_margin, x1 : x2] = True
        elif (key % 12) == 2:
            # RE (D)
            if (key != self.first_key):
                x1 = 0
                x2 = int(self.black_keys_width / 2) + self.keys_margin - self.black_keys_uncentering - int(self.keys_margin / 2)
                key_img_mask[0 : self.black_keys_height + self.keys_margin, x1 : x2] = True
            if (key != self.last_key):
                x1 = self.white_keys_width - int(self.black_keys_width / 2) - self.keys_margin + self.black_keys_uncentering + int(self.keys_margin / 2)
                x2 = self.white_keys_width
                key_img_mask[0 : self.black_keys_height + self.keys_margin, x1 : x2] = True
        elif (key % 12) == 7:
            # SOL (G)
            if (key != self.first_key):
                x1 = 0
                x2 = int(self.black_keys_width / 2) + self.keys_margin - self.black_keys_uncentering - int(self.keys_margin / 2)
                key_img_mask[0 : self.black_keys_height + self.keys_margin, x1 : x2] = True
            if (key != self.last_key):
                x1 = self.white_keys_width - int(self.black_keys_width / 2) - self.keys_margin + int(self.keys_margin / 2)
                x2 = self.white_keys_width
                key_img_mask[0 : self.black_keys_height + self.keys_margin, x1 : x2] = True
        elif (key % 12) == 9:
            # LA (A)
            if (key != self.first_key):
                x1 = 0
                x2 = int(self.black_keys_width / 2) + self.keys_margin - int(self.keys_margin / 2)
                key_img_mask[0 : self.black_keys_height + self.keys_margin, x1 : x2] = True
            if (key != self.last_key):
                x1 = self.white_keys_width - int(self.black_keys_width / 2) - self.keys_margin + self.black_keys_uncentering + int(self.keys_margin / 2)
                x2 = self.white_keys_width
                key_img_mask[0 : self.black_keys_height + self.keys_margin, x1 : x2] = True
        elif (key % 12) in [4, 11]:
            # MI or SI (E or B)
            if (key != self.first_key):
                x1 = 0
                x2 = int(self.black_keys_width / 2) + self.keys_margin + self.black_keys_uncentering - int(self.keys_margin / 2)
                key_img_mask[0 : self.black_keys_height + self.keys_margin, x1 : x2] = True
        return key_img_mask

    def drawWhiteKey(self, key, pressed=False, white_shadow=False, black_shadow=False):
        key_img = np.ones((self.white_keys_height, self.white_keys_width, 3), dtype=np.uint8) * 255
        for px in range(0, self.white_keys_width):
            x = (px + 0.5) - (self.white_keys_width / 2.0)
            for py in range(0, int(self.white_keys_width / 2)):
                y = py - self.white_keys_width / 0.75
                if np.sqrt(x**2 + y**2) > (self.white_keys_width / 0.74):
                    key_img[self.white_keys_height -1 - py, px, :] = np.array([0, 0, 0], dtype=np.uint8)
        key_img_mask = self.getWhiteKeyMask(key)
        return key_img, key_img_mask

    def drawBlackKey(self, key, pressed=False):
        key_img = np.ones((self.black_keys_height, self.black_keys_width, 3), dtype=np.uint8)
        key_img *= np.linspace(0, 50, self.black_keys_height).astype(np.uint8)[:, None, None]
        for px in range(0, self.black_keys_width):
            x = (px + 0.5) - (self.black_keys_width / 2.0)
            for py in range(0, self.black_keys_width):
                y = py
                if np.sqrt(x**2 + y**2) < (self.black_keys_width / 2.0):
                    key_img[-py, px, :] = (75, 75, 75)
        return key_img

    def drawKey(self, img, key, pressed=False, white_shadow=False, black_shadow=False, color=None):
        xstart = self.getXStart(key)
        ystart = self.red_band_height
        if self.isWhiteKey(key):
            img_key, mask = self.drawWhiteKey(key, pressed, white_shadow, black_shadow)
            if color is not None:
                img_key = self.addColor(img_key, color, key_weight=0.25, color_weight=1)
            img_temp = img[ystart : ystart + self.white_keys_height, xstart : xstart + self.white_keys_width]
            img_temp = np.where(np.logical_not(mask)[:, :, None], img_key, img_temp)
            img[ystart : ystart + self.white_keys_height, xstart : xstart + self.white_keys_width] = img_temp
        else:
            img_key = self.drawBlackKey(key, pressed)
            if color is not None:
                img_key = self.addColor(img_key, color, key_weight=0.5, color_weight=1)
            img[ystart : ystart + self.black_keys_height, xstart : xstart + self.black_keys_width, :] = img_key

    def drawKeyboard(self):
        self.img_keyboard = np.ones((self.img_height, self.img_width, 3), dtype=np.uint8)
        self.img_keyboard *= self.background_color
        self.img_keyboard[0 : self.red_band_height] = self.red_band_color
        for key in self.keysIterator():
            self.drawKey(self.img_keyboard, key)
        self.img_keyboard.flags.writeable = False

    def draw(self, pressed_keys=None, colors=None):
        img_keyboard_out = np.copy(self.img_keyboard)
        for k, key in enumerate(pressed_keys):
            ws_key = 1 if ((key % 12) in [0, 5]) else 2
            w_shadow = (key - ws_key) not in pressed_keys
            b_shadow = ((key % 12) in [2, 4, 7, 9, 11]) and ((key - 1) not in pressed_keys)
            color = (255, 0, 0) if colors is None else colors[k]
            self.drawKey(img_keyboard_out, key, pressed=True, white_shadow=w_shadow, black_shadow=b_shadow, color=color)
        return img_keyboard_out


class PianoVideoFinggering:
    def __init__(self, keyboard, fps, monitor_duration, start_offset, end_offset, height_ratio):
        self.fps = fps # Frames / seconds
        self.monitor_duration = monitor_duration # Seconds
        self.start_offset = start_offset # Seconds
        self.end_offset = end_offset # Seconds
        self.height_ratio = height_ratio # No unit
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.piano_model = PianoModel(keyboard=keyboard)
        self.track_colors = [(0, 255, 255), (0, 0, 255), (153, 255, 51), (0, 204, 0)]

    def drawFrame(self, t, tracks):
        frame = np.copy(self.background)
        keys_pressed = []
        keys_color = []
        for k_track, track in enumerate(tracks):
            color = self.track_colors[k_track]
            for note in track:
                t1 = note["starting"]
                t2 = note["starting"] + 0.95 * note["duration"]
                key = note["key"]
                if (key < self.piano_model.first_key) or (key > self.piano_model.last_key):
                    continue
                t1_inside = (t1 >= t) and (t1 <= (t + self.monitor_duration))
                t2_inside = (t2 >= t) and (t2 <= (t + self.monitor_duration))
                t12_over = (t1 <= t) and (t2 >= (t + self.monitor_duration))
                if t1_inside or t2_inside or t12_over:
                    xmin, xmax = self.piano_model.getCoordinates(key)
                    ymin = self.monitor_height - int((t2 - t) * ((self.monitor_height - 1) / self.monitor_duration))
                    ymax = self.monitor_height - int((t1 - t) * ((self.monitor_height - 1) / self.monitor_duration))
                    ymin = max(0, ymin)
                    ymax = min(self.monitor_height - 1, ymax)
                    frame[ymin : ymax, xmin : xmax, :] = np.array(color, dtype=np.uint8)
                if (t >= t1) and (t <= t2):
                    keys_pressed.append(key)
                    keys_color.append(color)

        img_piano = self.piano_model.draw(keys_pressed, keys_color)
        frame[-self.piano_height:, :, :] = img_piano
        return frame

    def createBackground(self):
        self.background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        color_start = np.array([153, 0, 76], dtype=np.uint8)
        color_stop = np.array([0, 0, 0], dtype=np.uint8)
        a = (color_stop.astype(float) - color_start.astype(float)) / (self.monitor_height - 1)
        b = color_start.astype(float)
        degrade_color = np.zeros((self.monitor_height, 3), dtype=np.uint8)
        for k in range(0, self.monitor_height):
            color = k * a + b
            degrade_color[k, :] = color.astype(np.uint8)
        self.background[0 : self.monitor_height, :, :] = degrade_color[: ,None, : ]

    def process(self, in_midi_path, out_video_path):
        # Extract midi info
        loader = MidiLoader()
        tracks = loader.extract(in_midi_path)
        # Initialize output video
        ph, pw = self.piano_model.getShape()
        self.piano_height = ph
        self.width = pw
        self.height = int(round(self.width * self.height_ratio))
        self.monitor_height = self.height - self.piano_height
        out_vid = cv2.VideoWriter(out_video_path, self.fourcc, self.fps, (self.width, self.height))
        # Find the duration of the music
        time_max_s = -1
        for track in tracks:
            for note in track:
                time_note_end_s = note["starting"] + note["duration"]
                if(time_note_end_s > time_max_s):
                    time_max_s = time_note_end_s
        self.createBackground()
        # Drawing frames
        nb_frames = int(np.ceil((time_max_s + self.start_offset + self.end_offset) * self.fps))
        for k_frame in range(0, nb_frames):
            print("\rFrame n° %d / %d " % (k_frame, nb_frames), end="\r")
            t = (k_frame / self.fps) - self.start_offset
            frame = self.drawFrame(t, tracks)
            # Convert RGB to BGR
            frame = np.flip(frame, axis=2)
            out_vid.write(frame)
        out_vid.release()


if __name__ == '__main__':
    args = get_args()
    print("\nINPUT FILE: %s" % args.input)
    print("\nOUTPUT FILE: %s" % args.output)
    print(args)
    finger = PianoVideoFinggering(  keyboard=args.keyboard,
                                    fps=args.fps,
                                    monitor_duration=args.monitor_duration,
                                    start_offset=args.start_offset,
                                    end_offset=args.end_offset,
                                    height_ratio=args.height_ratio)
    finger.process(args.input, args.output)
