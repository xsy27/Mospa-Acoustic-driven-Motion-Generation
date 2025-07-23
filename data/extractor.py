import librosa
import numpy as np

class FeatureExtractor:
    @staticmethod
    def get_melspectrogram(audio, sample_rate, n_fft, hop_length, window):
        melspe = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, window=window)
        melspe_db = librosa.power_to_db(melspe, ref=np.max)
        # print(f'{melspe_db.shape} -> melspe_db')
        return melspe_db

    @staticmethod
    def get_hpss(audio, n_fft, hop_length, window):
        audio_harmonic, audio_percussive = librosa.effects.hpss(audio, n_fft=n_fft, hop_length=hop_length, window=window)
        # print(f'{audio_percussive.shape} -> audio_percussive')
        return audio_harmonic, audio_percussive

    @staticmethod
    def get_mfcc(melspe_db):
        mfcc = librosa.feature.mfcc(S=melspe_db)
        # print(f'{mfcc.shape} -> mfcc')
        return mfcc

    @staticmethod
    def get_mfcc_delta(mfcc):
        mfcc_delta = librosa.feature.delta(mfcc, width=3)
        # print(f'{mfcc_delta.shape} -> mfcc_delta')
        return mfcc_delta

    @staticmethod
    def get_mfcc_delta2(mfcc):
        mfcc_delta_delta = librosa.feature.delta(mfcc, width=3, order=2)
        # print(f'{mfcc_delta_delta.shape} -> mfcc_delta_delta')
        return mfcc_delta_delta

    @staticmethod
    def get_harmonic_melspe_db(audio_harmonic, sr, n_fft, hop_length, window):
        harmonic_melspe = librosa.feature.melspectrogram(y=audio_harmonic, sr=sr, n_fft=n_fft, hop_length=hop_length, window=window)
        harmonic_melspe_db = librosa.power_to_db(harmonic_melspe, ref=np.max)
        # print(f'{harmonic_melspe_db.shape} -> harmonic_melspe_db')
        return harmonic_melspe_db

    @staticmethod
    def get_percussive_melspe_db(audio_percussive, sr):
        percussive_melspe = librosa.feature.melspectrogram(y=audio_percussive, sr=sr)
        percussive_melspe_db = librosa.power_to_db(percussive_melspe, ref=np.max)
        # print(f'{percussive_melspe_db.shape} -> percussive_melspe_db')
        return percussive_melspe_db

    @staticmethod
    def get_chroma_cqt(audio_harmonic, sr, n_fft, hop_length, octave=7):
        chroma_cqt_harmonic = librosa.feature.chroma_cqt(y=audio_harmonic, sr=sr, n_octaves=octave, hop_length=hop_length)
        # print(f'{chroma_cqt_harmonic.shape} -> chroma_cqt_harmonic')
        return chroma_cqt_harmonic

    @staticmethod
    def get_chroma_stft(audio_harmonic, sr, n_fft, hop_length, window):
        chroma_stft_harmonic = librosa.feature.chroma_stft(y=audio_harmonic, sr=sr, n_fft=n_fft, hop_length=hop_length, window=window)
        # print(f'{chroma_stft_harmonic.shape} -> chroma_stft_harmonic')
        return chroma_stft_harmonic

    @staticmethod
    def get_tonnetz(audio_harmonic, sr):
        tonnetz = librosa.feature.tonnetz(y=audio_harmonic, sr=sr)
        # print(f'{tonnetz.shape} -> tonnetz')
        return tonnetz

    @staticmethod
    def get_onset_strength(audio_percussive, sr, n_fft, hop_length, window):
        onset_env = librosa.onset.onset_strength(y=audio_percussive, sr=sr, aggregate=np.median, n_fft=n_fft, hop_length=hop_length, window=window)
        # print(f'{onset_env.reshape(1, -1).shape} -> onset_env')
        return onset_env

    @staticmethod
    def get_tempogram(onset_env, sr, hop_length, window):
        win_length = int(8.9 * sr / hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=win_length, window=window)
        # print(f'{tempogram.shape} -> tempogram')
        return tempogram

    @staticmethod
    def get_onset_beat(onset_env, sr, hop_length):
        onset_tempo, onset_beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
        beats_one_hot = np.zeros(len(onset_env))
        peaks_one_hot = np.zeros(len(onset_env))
        for idx in onset_beats:
            beats_one_hot[idx] = 1
        for idx in peaks:
            peaks_one_hot[idx] = 1

        beats_one_hot = beats_one_hot.reshape(1, -1)
        peaks_one_hot = peaks_one_hot.reshape(1, -1)

        # print(f'{beats_one_hot.shape} -> beats_feature')
        return beats_one_hot, peaks_one_hot
    
    @staticmethod
    def get_rms_energy(audio, n_fft, hop_length, window):
        S, phase = librosa.magphase(librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window=window))
        rms_energy = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)
        # print(f'{rms_energy.shape} -> rms_energy_feature')
        return rms_energy
    
    @staticmethod
    def get_binaural_stft(audio_l, audio_r, n_fft, hop_length, window):
        D_l = librosa.stft(y=audio_l, n_fft=n_fft, hop_length=hop_length, window=window)
        D_r = librosa.stft(y=audio_r, n_fft=n_fft, hop_length=hop_length, window=window)
        return D_l, D_r
    
    @staticmethod
    def get_binaural_mag_phase(D_l, D_r):
        magnitude_l, phase_l = librosa.magphase(D=D_l)
        magnitude_r, phase_r = librosa.magphase(D=D_r)
        return magnitude_l, magnitude_r, phase_l, phase_r
    
    @staticmethod
    def get_binaural_mean_mag(mag_l, mag_r):
        mean_mag = np.mean(np.array([mag_l, mag_r]), axis=0)
        return mean_mag
    
    @staticmethod
    def get_IPD(phase_l, phase_r):
        angle_l = np.angle(phase_l)
        angle_r = np.angle(phase_r)

        phase_diff = angle_r - angle_l

        sin_IPD = np.sin(phase_diff)
        cos_IPD = np.cos(phase_diff)
        return sin_IPD, cos_IPD
    
    @staticmethod
    def get_ILD(mag_l, mag_r):
        L_l = 20 * np.log10(mag_l)
        L_r = 20 * np.log10(mag_r)

        ILD = L_r - L_l

        return ILD