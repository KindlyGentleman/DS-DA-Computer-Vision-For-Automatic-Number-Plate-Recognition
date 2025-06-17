#!/usr/bin/env python
# realtime_plate_ocr_refactored.py
#
# Description:
# This script performs real-time license plate detection and optical character recognition (OCR)
# on a video stream (from a camera or file). It then identifies the plate's region of origin
# in Indonesia using a detailed city code dictionary and determines if the plate number is odd or even.
#
# Dependencies:
#   - Python ≥ 3.9
#   - Ultralytics YOLO ≥ 8.2 (`pip install ultralytics`)
#   - Fast Plate OCR ≥ 0.3 (`pip install fast-plate-ocr`)
#   - OpenCV (`pip install opencv-python`)
#   - NumPy (`pip install numpy`)
#
# How to Run:
#   - Place your trained YOLO model (e.g., license_plate_detector.pt) in a 'models' subfolder.
#   - Run from the terminal:
#     python realtime_plate_ocr_refactored.py --source 0  (for webcam)
#     python realtime_plate_ocr_refactored.py --source /path/to/your/video.mp4
#

# --- Standard Library Imports ---
from __future__ import annotations  # Enables postponed evaluation of type annotations
import argparse
import re
import signal
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

# --- Third-Party Library Imports ---
import cv2
import numpy as np
from ultralytics import YOLO          # For object detection
from fast_plate_ocr import ONNXPlateRecognizer # For OCR

# -----------------------------------------------------------------------------
# DETAILED INDONESIAN CITY/REGION CODE DICTIONARY
# -----------------------------------------------------------------------------
# This dictionary maps license plate prefixes (e.g., "B", "D", "AG") and the
# first letter of their suffix to a specific city or regency in Indonesia.
# The "default" key provides a general region name if a specific suffix is not found.
#
# Structure:
# {
#   "PREFIX": {
#     "default": "General Area Name",
#     "SUFFIX_CHAR_1": "Specific City/Regency Name",
#     "SUFFIX_CHAR_2": "Another Specific City/Regency Name",
#     ...
#   }
# }
# -----------------------------------------------------------------------------
detailed_city_code_dict = {
    # Sumatera
    "BL": {"default": "Nanggroe Aceh Darussalam", "A": "Kota Banda Aceh", "J": "Kota Banda Aceh", "B": "Gayo Lues", "C": "Aceh Barat Daya", "D": "Aceh Timur", "E": "Aceh Barat", "F": "Kota Langsa", "G": "Aceh Tengah", "H": "Aceh Tenggara", "I": "Kota Subulussalam", "K": "Aceh Utara", "Q": "Aceh Utara", "L": "Aceh Besar", "M": "Kota Sabang", "N": "Kota Lhokseumawe", "O": "Pidie Jaya", "P": "Pidie", "R": "Aceh Singkil", "S": "Simeulue", "T": "Aceh Selatan", "U": "Aceh Tamiang", "V": "Nagan Raya", "W": "Aceh Jaya", "Y": "Bener Meriah", "Z": "Bireuen"},
    "BB": {"default": "Sumatera Utara (Tapanuli)", "A": "Kota Sibolga", "L": "Kota Sibolga", "N": "Kota Sibolga", "B": "Tapanuli Utara", "C": "Samosir", "D": "Humbang Hasundutan", "E": "Toba", "F": "Kota Padang Sidempuan", "H": "Kota Padang Sidempuan", "G": "Tapanuli Selatan", "J": "Padang Lawas Utara", "K": "Padang Lawas", "M": "Tapanuli Tengah", "Q": "Nias Utara", "R": "Mandailing Natal", "T": "Kota Gunungsitoli", "U": "Nias Barat", "V": "Nias", "W": "Nias Selatan", "Y": "Dairi", "Z": "Pakpak Bharat"},
    "BK": {"default": "Sumatera Utara (Pesisir Timur)", "A": "Kota Medan", "B": "Kota Medan", "C": "Kota Medan", "D": "Kota Medan", "E": "Kota Medan", "F": "Kota Medan", "G": "Kota Medan", "H": "Kota Medan", "I": "Kota Medan", "K": "Kota Medan", "L": "Kota Medan", "J": "Labuhanbatu Utara", "M": "Deli Serdang", "N": "Kota Tebing Tinggi", "O": "Batubara", "P": "Langkat", "Q": "Kota Tanjung Balai", "R": "Kota Binjai", "S": "Karo", "T": "Simalungun", "U": "Simalungun", "V": "Asahan", "W": "Kota Pematang Siantar", "X": "Serdang Bedagai", "Y": "Labuhanbatu", "Z": "Labuhanbatu Selatan"},
    "BA": {"default": "Sumatera Barat", "A": "Kota Padang", "B": "Kota Padang", "O": "Kota Padang", "Q": "Kota Padang", "C": "Lima Puluh Kota", "X": "Lima Puluh Kota", "D": "Pasaman", "E": "Tanah Datar", "F": "Padang Pariaman", "G": "Pesisir Selatan", "I": "Pesisir Selatan", "H": "Solok", "J": "Kota Sawahlunto", "K": "Sijunjung", "L": "Kota Bukittinggi", "M": "Kota Payakumbuh", "N": "Kota Padang Panjang", "P": "Kota Solok", "S": "Pasaman Barat", "T": "Agam", "Z": "Agam", "U": "Kepulauan Mentawai", "V": "Dharmasraya", "W": "Kota Pariaman", "Y": "Solok Selatan"},
    "BM": {"default": "Riau", "A": "Kota Pekanbaru", "J": "Kota Pekanbaru", "N": "Kota Pekanbaru", "O": "Kota Pekanbaru", "Q": "Kota Pekanbaru", "T": "Kota Pekanbaru", "V": "Kota Pekanbaru", "B": "Indragiri Hulu", "C": "Pelalawan", "I": "Pelalawan", "D": "Bengkalis", "E": "Bengkalis", "F": "Kampar", "Z": "Kampar", "G": "Indragiri Hilir", "H": "Kota Dumai", "R": "Kota Dumai", "K": "Kuantan Singingi", "X": "Kuantan Singingi/Kep. Meranti", "M": "Rokan Hulu", "U": "Rokan Hulu", "P": "Rokan Hilir", "W": "Rokan Hilir", "S": "Siak", "Y": "Siak"},
    "BH": {"default": "Jambi", "A": "Kota Jambi", "H": "Kota Jambi", "M": "Kota Jambi", "N": "Kota Jambi", "Y": "Kota Jambi", "Z": "Kota Jambi", "B": "Batanghari", "V": "Batanghari", "C": "Tebo", "W": "Tebo", "D": "Kerinci", "E": "Tanjung Jabung Barat", "O": "Tanjung Jabung Barat", "F": "Merangin", "P": "Merangin", "X": "Merangin", "G": "Muaro Jambi", "I": "Muaro Jambi", "J": "Tanjung Jabung Timur", "T": "Tanjung Jabung Timur", "K": "Bungo", "U": "Bungo", "Q": "Sarolangun", "S": "Sarolangun", "R": "Kota Sungai Penuh"},
    "BG": {"default": "Sumatera Selatan", "A": "Kota Palembang", "I": "Kota Palembang", "M": "Kota Palembang", "N": "Kota Palembang", "O": "Kota Palembang", "U": "Kota Palembang", "X": "Kota Palembang", "Z": "Kota Palembang", "B": "Musi Banyuasin", "C": "Kota Prabumulih", "D": "Muara Enim", "E": "Lahat", "F": "Ogan Komering Ulu", "G": "Musi Rawas", "H": "Kota Lubuk Linggau", "J": "Banyuasin", "R": "Banyuasin", "K": "Ogan Komering Ilir", "P": "Penukal Abab Lematang Ilir", "Q": "Musi Rawas Utara", "S": "Empat Lawang", "T": "Ogan Ilir", "V": "Ogan Komering Ulu Selatan", "W": "Kota Pagaralam", "Y": "Ogan Komering Ulu Timur"},
    "BD": {"default": "Bengkulu", "A": "Kota Bengkulu", "C": "Kota Bengkulu", "E": "Kota Bengkulu", "I": "Kota Bengkulu", "U": "Kota Bengkulu", "V": "Kota Bengkulu", "B": "Bengkulu Selatan", "M": "Bengkulu Selatan", "D": "Bengkulu Utara", "Q": "Bengkulu Utara", "S": "Bengkulu Utara", "F": "Rejang Lebong", "K": "Rejang Lebong", "G": "Kepahiang", "H": "Lebong", "N": "Muko Muko", "T": "Muko Muko", "P": "Seluma", "R": "Seluma", "W": "Kaur", "Y": "Bengkulu Tengah"},
    "BE": {"default": "Lampung", "A": "Kota Bandar Lampung", "B": "Kota Bandar Lampung", "C": "Kota Bandar Lampung", "D": "Lampung Selatan", "E": "Lampung Selatan", "O": "Lampung Selatan", "F": "Kota Metro", "G": "Lampung Tengah", "H": "Lampung Tengah", "I": "Lampung Tengah", "J": "Lampung Utara", "K": "Lampung Utara", "L": "Mesuji", "M": "Lampung Barat", "N": "Lampung Timur", "P": "Lampung Timur", "Q": "Tulang Bawang Barat", "R": "Pesawaran", "S": "Tulang Bawang", "T": "Tulang Bawang", "U": "Pringsewu", "V": "Tanggamus", "Z": "Tanggamus", "W": "Way Kanan", "X": "Pesisir Barat"},
    "BN": {"default": "Kep. Bangka Belitung", "A": "Kota Pangkal Pinang", "P": "Kota Pangkal Pinang", "B": "Bangka", "Q": "Bangka", "C": "Bangka Tengah", "T": "Bangka Tengah", "D": "Bangka Barat", "R": "Bangka Barat", "E": "Bangka Selatan", "V": "Bangka Selatan", "F": "Belitung", "W": "Belitung", "G": "Belitung Timur", "X": "Belitung Timur"},
    "BP": {"default": "Kepulauan Riau", "A": "Kota Tanjung Pinang", "P": "Kota Tanjung Pinang", "T": "Kota Tanjung Pinang", "W": "Kota Tanjung Pinang", "B": "Bintan", "C": "Kota Batam", "D": "Kota Batam", "E": "Kota Batam", "F": "Kota Batam", "G": "Kota Batam", "H": "Kota Batam", "I": "Kota Batam", "J": "Kota Batam", "M": "Kota Batam", "O": "Kota Batam", "Q": "Kota Batam", "R": "Kota Batam", "U": "Kota Batam", "V": "Kota Batam", "X": "Kota Batam", "Z": "Kota Batam", "K": "Karimun", "L": "Lingga", "N": "Natuna", "S": "Kepulauan Anambas"},
    # Jawa & Banten
    "B":  {"default": "DKI Jakarta/Sekitarnya", "B": "Jakarta Barat", "H": "Jakarta Barat", "P": "Jakarta Pusat", "S": "Jakarta Selatan", "D": "Jakarta Selatan", "T": "Jakarta Timur", "R": "Jakarta Timur", "U": "Jakarta Utara", "E": "Kota Depok/Kab. Bogor", "F": "Kabupaten Bekasi", "K": "Kota Bekasi", "Z": "Kota Depok (Cinere)", "J": "Kab. Tangerang (Kelapa Dua)", "C": "Kota Tangerang (Cikokol)", "V": "Kota Tangerang (Ciledug)", "N": "Kota Tangerang Selatan (Serpong)", "W": "Kota Tangerang Selatan (Ciputat)"},
    "A":  {"default": "Banten", "A": "Kota Serang", "B": "Kota Serang", "C": "Kota Serang", "D": "Kota Serang", "E": "Kabupaten Serang", "F": "Kabupaten Serang", "G": "Kabupaten Serang", "H": "Kabupaten Serang", "I": "Kabupaten Serang", "J": "Pandeglang", "K": "Pandeglang", "L": "Pandeglang", "M": "Pandeglang", "N": "Lebak", "O": "Lebak", "P": "Lebak", "Q": "Lebak", "R": "Kota Cilegon", "S": "Kota Cilegon", "T": "Kota Cilegon", "U": "Kota Cilegon", "V": "Kab. Tangerang (Balaraja)", "W": "Kab. Tangerang (Balaraja)", "X": "Kab. Tangerang (Balaraja)", "Y": "Kab. Tangerang (Balaraja)", "Z": "Kab. Tangerang (Balaraja)"},
    "D":  {"default": "Bandung Raya", "A": "Kota Bandung", "B": "Kota Bandung", "C": "Kota Bandung", "D": "Kota Bandung", "E": "Kota Bandung", "F": "Kota Bandung", "G": "Kota Bandung", "H": "Kota Bandung", "I": "Kota Bandung", "J": "Kota Bandung", "K": "Kota Bandung", "L": "Kota Bandung", "M": "Kota Bandung", "N": "Kota Bandung", "O": "Kota Bandung", "P": "Kota Bandung", "Q": "Kota Bandung", "R": "Kota Bandung", "S": "Kota Cimahi", "T": "Kota Cimahi", "U": "Bandung Barat", "X": "Bandung Barat", "V": "Kabupaten Bandung", "W": "Kabupaten Bandung", "Y": "Kabupaten Bandung", "Z": "Kabupaten Bandung"},
    "E":  {"default": "Eks Keresidenan Cirebon", "A": "Kota Cirebon", "B": "Kota Cirebon", "C": "Kota Cirebon", "D": "Kota Cirebon", "E": "Kota Cirebon", "F": "Kota Cirebon", "G": "Kota Cirebon", "H": "Kabupaten Cirebon", "I": "Kabupaten Cirebon", "J": "Kabupaten Cirebon", "K": "Kabupaten Cirebon", "L": "Kabupaten Cirebon", "M": "Kabupaten Cirebon", "N": "Kabupaten Cirebon", "O": "Kabupaten Cirebon", "P": "Indramayu", "Q": "Indramayu", "R": "Indramayu", "S": "Indramayu", "T": "Indramayu", "U": "Majalengka", "V": "Majalengka", "W": "Majalengka", "X": "Majalengka", "Y": "Kuningan", "Z": "Kuningan"},
    "F":  {"default": "Eks Keresidenan Bogor", "A": "Kota Bogor", "B": "Kota Bogor", "C": "Kota Bogor", "D": "Kota Bogor", "E": "Kota Bogor", "F": "Kabupaten Bogor", "G": "Kabupaten Bogor", "H": "Kabupaten Bogor", "I": "Kabupaten Bogor", "J": "Kabupaten Bogor", "K": "Kabupaten Bogor", "L": "Kabupaten Bogor", "M": "Kabupaten Bogor", "N": "Kabupaten Bogor", "P": "Kabupaten Bogor", "R": "Kabupaten Bogor", "O": "Kota Sukabumi", "S": "Kota Sukabumi", "T": "Kota Sukabumi", "Q": "Kabupaten Sukabumi", "U": "Kabupaten Sukabumi", "V": "Kabupaten Sukabumi", "W": "Cianjur", "X": "Cianjur", "Y": "Cianjur", "Z": "Cianjur"},
    "T":  {"default": "Eks Keresidenan Karawang", "A": "Purwakarta", "B": "Purwakarta", "C": "Purwakarta", "I": "Purwakarta", "J": "Purwakarta", "D": "Karawang", "E": "Karawang", "F": "Karawang", "G": "Karawang", "H": "Karawang", "K": "Karawang", "L": "Karawang", "M": "Karawang", "N": "Karawang", "O": "Karawang", "P": "Karawang", "Q": "Karawang", "R": "Karawang", "S": "Karawang", "T": "Subang", "U": "Subang", "V": "Subang", "W": "Subang", "X": "Subang", "Y": "Subang", "Z": "Subang"},
    "Z":  {"default": "Eks Keresidenan Priangan Timur", "A": "Sumedang", "B": "Sumedang", "C": "Sumedang", "D": "Garut", "E": "Garut", "F": "Garut", "G": "Garut", "H": "Kota Tasikmalaya", "I": "Kota Tasikmalaya", "J": "Kota Tasikmalaya", "K": "Kota Tasikmalaya", "L": "Kota Tasikmalaya", "M": "Kota Tasikmalaya", "N": "Kabupaten Tasikmalaya", "O": "Kabupaten Tasikmalaya", "P": "Kabupaten Tasikmalaya", "Q": "Kabupaten Tasikmalaya", "R": "Kabupaten Tasikmalaya", "S": "Kabupaten Tasikmalaya", "T": "Ciamis", "V": "Ciamis", "W": "Ciamis", "U": "Pangandaran", "X": "Kota Banjar", "Y": "Kota Banjar", "Z": "Kota Banjar"},
    "H":  {"default": "Eks Keresidenan Semarang", "A": "Kota Semarang", "F": "Kota Semarang", "G": "Kota Semarang", "H": "Kota Semarang", "P": "Kota Semarang", "Q": "Kota Semarang", "R": "Kota Semarang", "S": "Kota Semarang", "W": "Kota Semarang", "Y": "Kota Semarang", "Z": "Kota Semarang", "B": "Kota Salatiga", "K": "Kota Salatiga", "O": "Kota Salatiga", "T": "Kota Salatiga", "C": "Kabupaten Semarang", "I": "Kabupaten Semarang", "L": "Kabupaten Semarang", "V": "Kabupaten Semarang", "D": "Kendal", "M": "Kendal", "U": "Kendal", "E": "Demak", "J": "Demak", "N": "Demak"},
    "G":  {"default": "Eks Keresidenan Pekalongan", "A": "Kota Pekalongan", "H": "Kota Pekalongan", "S": "Kota Pekalongan", "B": "Kabupaten Pekalongan", "K": "Kabupaten Pekalongan", "O": "Kabupaten Pekalongan", "T": "Kabupaten Pekalongan", "C": "Batang", "L": "Batang", "V": "Batang", "D": "Pemalang", "I": "Pemalang", "M": "Pemalang", "W": "Pemalang", "E": "Kota Tegal", "N": "Kota Tegal", "Y": "Kota Tegal", "F": "Kabupaten Tegal", "P": "Kabupaten Tegal", "Q": "Kabupaten Tegal", "Z": "Kabupaten Tegal", "G": "Brebes", "J": "Brebes", "R": "Brebes", "U": "Brebes"},
    "K":  {"default": "Eks Keresidenan Pati", "A": "Pati", "G": "Pati", "H": "Pati", "S": "Pati", "U": "Pati", "B": "Kudus", "K": "Kudus", "O": "Kudus", "R": "Kudus", "T": "Kudus", "C": "Jepara", "L": "Jepara", "Q": "Jepara", "V": "Jepara", "D": "Rembang", "I": "Rembang", "M": "Rembang", "W": "Rembang", "E": "Blora", "N": "Blora", "Y": "Blora", "F": "Grobogan", "J": "Grobogan", "P": "Grobogan", "Z": "Grobogan"},
    "R":  {"default": "Eks Keresidenan Banyumas", "A": "Banyumas", "E": "Banyumas", "G": "Banyumas", "H": "Banyumas", "J": "Banyumas", "R": "Banyumas", "S": "Banyumas", "B": "Cilacap", "F": "Cilacap", "K": "Cilacap", "N": "Cilacap", "P": "Cilacap", "T": "Cilacap", "C": "Purbalingga", "L": "Purbalingga", "Q": "Purbalingga", "U": "Purbalingga", "V": "Purbalingga", "Z": "Purbalingga", "D": "Banjarnegara", "I": "Banjarnegara", "M": "Banjarnegara", "O": "Banjarnegara", "W": "Banjarnegara", "Y": "Banjarnegara"},
    "AA": {"default": "Eks Keresidenan Kedu", "A": "Kota Magelang", "H": "Kota Magelang", "S": "Kota Magelang", "U": "Kota Magelang", "B": "Kabupaten Magelang", "G": "Kabupaten Magelang", "K": "Kabupaten Magelang", "O": "Kabupaten Magelang", "T": "Kabupaten Magelang", "C": "Purworejo", "L": "Purworejo", "Q": "Purworejo", "V": "Purworejo", "D": "Kebumen", "J": "Kebumen", "M": "Kebumen", "W": "Kebumen", "E": "Temanggung", "N": "Temanggung", "Y": "Temanggung", "F": "Wonosobo", "P": "Wonosobo", "Z": "Wonosobo"},
    "AD": {"default": "Eks Keresidenan Surakarta", "A": "Kota Surakarta", "H": "Kota Surakarta", "S": "Kota Surakarta", "U": "Kota Surakarta", "B": "Sukoharjo", "K": "Sukoharjo", "O": "Sukoharjo", "T": "Sukoharjo", "C": "Klaten", "J": "Klaten", "L": "Klaten", "Q": "Klaten", "V": "Klaten", "D": "Boyolali", "M": "Boyolali", "W": "Boyolali", "E": "Sragen", "N": "Sragen", "Y": "Sragen", "F": "Karanganyar", "P": "Karanganyar", "Z": "Karanganyar", "G": "Wonogiri", "I": "Wonogiri", "R": "Wonogiri"},
    "AB": {"default": "DI Yogyakarta", "A": "Kota Yogyakarta", "F": "Kota Yogyakarta", "H": "Kota Yogyakarta", "I": "Kota Yogyakarta", "S": "Kota Yogyakarta", "B": "Bantul", "G": "Bantul", "J": "Bantul", "K": "Bantul", "T": "Bantul", "C": "Kulon Progo", "L": "Kulon Progo", "O": "Kulon Progo", "P": "Kulon Progo", "V": "Kulon Progo", "D": "Gunungkidul", "M": "Gunungkidul", "R": "Gunungkidul", "W": "Gunungkidul", "E": "Sleman", "N": "Sleman", "Q": "Sleman", "U": "Sleman", "X": "Sleman", "Y": "Sleman", "Z": "Sleman"},
    "L":  {"default": "Kota Surabaya"},
    "M":  {"default": "Eks Keresidenan Madura", "A": "Pamekasan", "B": "Pamekasan", "C": "Pamekasan", "D": "Pamekasan", "E": "Pamekasan", "F": "Pamekasan", "G": "Bangkalan", "H": "Bangkalan", "I": "Bangkalan", "J": "Bangkalan", "K": "Bangkalan", "L": "Bangkalan", "M": "Bangkalan", "N": "Sampang", "O": "Sampang", "P": "Sampang", "Q": "Sampang", "R": "Sampang", "S": "Sampang", "T": "Sumenep", "U": "Sumenep", "V": "Sumenep", "W": "Sumenep", "X": "Sumenep", "Y": "Sumenep", "Z": "Sumenep"},
    "N":  {"default": "Eks Keresidenan Malang-Pasuruan", "A": "Kota Malang", "B": "Kota Malang", "C": "Kota Malang", "D": "Kota Malang", "E": "Kabupaten Malang", "F": "Kabupaten Malang", "G": "Kabupaten Malang", "H": "Kabupaten Malang", "I": "Kabupaten Malang", "J": "Kota Batu", "K": "Kota Batu", "L": "Kota Batu", "M": "Kabupaten Probolinggo", "N": "Kabupaten Probolinggo", "O": "Kabupaten Probolinggo", "P": "Kota Probolinggo", "Q": "Kota Probolinggo", "R": "Kota Probolinggo", "S": "Lumajang", "T": "Kabupaten Pasuruan", "U": "Lumajang", "Y": "Lumajang", "Z": "Lumajang", "V": "Kota Pasuruan", "W": "Kota Pasuruan", "X": "Kota Pasuruan"},
    "P":  {"default": "Eks Keresidenan Besuki", "A": "Bondowoso", "B": "Bondowoso", "C": "Bondowoso", "D": "Situbondo", "E": "Situbondo", "F": "Situbondo", "G": "Jember", "H": "Jember", "I": "Jember", "J": "Jember", "K": "Jember", "L": "Jember", "M": "Jember", "N": "Jember", "O": "Jember", "P": "Jember", "Q": "Banyuwangi", "R": "Banyuwangi", "S": "Banyuwangi", "T": "Banyuwangi", "U": "Banyuwangi", "V": "Banyuwangi", "W": "Banyuwangi", "X": "Banyuwangi", "Y": "Banyuwangi", "Z": "Banyuwangi"},
    "S":  {"default": "Eks Keresidenan Bojonegoro", "A": "Bojonegoro", "B": "Bojonegoro", "C": "Bojonegoro", "D": "Bojonegoro", "E": "Tuban", "F": "Tuban", "G": "Tuban", "H": "Tuban", "I": "Tuban", "J": "Lamongan", "K": "Lamongan", "L": "Lamongan", "M": "Lamongan", "N": "Kabupaten Mojokerto", "P": "Kabupaten Mojokerto", "Q": "Kabupaten Mojokerto", "R": "Kabupaten Mojokerto", "O": "Jombang", "W": "Jombang", "X": "Jombang", "Y": "Jombang", "Z": "Jombang", "S": "Kota Mojokerto", "T": "Kota Mojokerto", "U": "Kota Mojokerto", "V": "Kota Mojokerto"},
    "W":  {"default": "Gresik & Sidoarjo", "A": "Gresik", "B": "Gresik", "C": "Gresik", "D": "Gresik", "E": "Gresik", "F": "Gresik", "G": "Gresik", "H": "Gresik", "I": "Gresik", "J": "Gresik", "K": "Gresik", "L": "Gresik", "M": "Gresik", "N": "Sidoarjo", "O": "Sidoarjo", "P": "Sidoarjo", "Q": "Sidoarjo", "R": "Sidoarjo", "S": "Sidoarjo", "T": "Sidoarjo", "U": "Sidoarjo", "V": "Sidoarjo", "W": "Sidoarjo", "X": "Sidoarjo", "Y": "Sidoarjo", "Z": "Sidoarjo"},
    "AE": {"default": "Eks Keresidenan Madiun", "A": "Kota Madiun", "B": "Kota Madiun", "C": "Kota Madiun", "D": "Kota Madiun", "E": "Kabupaten Madiun", "F": "Kabupaten Madiun", "G": "Kabupaten Madiun", "H": "Kabupaten Madiun", "I": "Kabupaten Madiun", "J": "Ngawi", "K": "Ngawi", "L": "Ngawi", "M": "Ngawi", "N": "Magetan", "O": "Magetan", "P": "Magetan", "Q": "Magetan", "R": "Magetan", "S": "Ponorogo", "T": "Ponorogo", "U": "Ponorogo", "V": "Ponorogo", "W": "Ponorogo", "X": "Pacitan", "Y": "Pacitan", "Z": "Pacitan"},
    "AG": {"default": "Eks Keresidenan Kediri", "A": "Kota Kediri", "B": "Kota Kediri", "C": "Kota Kediri", "D": "Kota Kediri", "E": "Kabupaten Kediri", "F": "Kabupaten Kediri", "G": "Kabupaten Kediri", "H": "Kabupaten Kediri", "J": "Kabupaten Kediri", "O": "Kabupaten Kediri", "I": "Kabupaten Blitar", "K": "Kabupaten/Kota Blitar", "L": "Kabupaten Blitar", "M": "Kabupaten Blitar", "P": "Kabupaten Blitar", "N": "Kota Blitar", "Q": "Kota Blitar", "R": "Tulungagung", "S": "Tulungagung", "T": "Tulungagung", "U": "Nganjuk", "V": "Nganjuk", "W": "Nganjuk", "X": "Nganjuk", "Y": "Trenggalek", "Z": "Trenggalek"},
    # Bali & Nusa Tenggara
    "DK": {"default": "Bali", "A": "Kota Denpasar", "B": "Kota Denpasar", "C": "Kota Denpasar", "D": "Kota Denpasar", "E": "Kota Denpasar", "I": "Kota Denpasar", "X": "Kota Denpasar", "F": "Badung", "J": "Badung", "O": "Badung", "Q": "Badung", "G": "Tabanan", "H": "Tabanan", "K": "Gianyar", "L": "Gianyar", "M": "Klungkung", "N": "Klungkung", "P": "Bangli", "R": "Bangli", "S": "Karangasem", "T": "Karangasem", "U": "Buleleng", "V": "Buleleng", "W": "Jembrana", "Z": "Jembrana"},
    "DR": {"default": "NTB (Lombok)", "A": "Kota Mataram", "B": "Kota Mataram", "C": "Kota Mataram", "E": "Kota Mataram", "F": "Kota Mataram", "N": "Kota Mataram", "O": "Kota Mataram", "P": "Kota Mataram", "R": "Kota Mataram", "X": "Kota Mataram", "D": "Lombok Utara", "G": "Lombok Utara", "M": "Lombok Utara", "H": "Lombok Barat", "J": "Lombok Barat", "K": "Lombok Barat", "T": "Lombok Barat", "W": "Lombok Barat", "L": "Lombok Timur", "Q": "Lombok Timur", "Y": "Lombok Timur", "S": "Lombok Tengah", "U": "Lombok Tengah", "V": "Lombok Tengah", "Z": "Lombok Tengah"},
    "EA": {"default": "NTB (Sumbawa)", "A": "Sumbawa", "C": "Sumbawa", "D": "Sumbawa", "E": "Sumbawa", "F": "Sumbawa", "P": "Sumbawa", "H": "Sumbawa Barat", "K": "Sumbawa Barat", "L": "Kota Bima", "S": "Kota Bima", "M": "Dompu", "N": "Dompu", "Q": "Dompu", "R": "Dompu", "T": "Dompu", "W": "Bima", "X": "Bima", "Y": "Bima", "Z": "Bima"},
    "DH": {"default": "NTT (Timor)", "A": "Kota Kupang", "H": "Kota Kupang", "K": "Kota Kupang", "B": "Kupang", "N": "Kupang", "C": "Timor Tengah Selatan", "D": "Timor Tengah Utara", "M": "Timor Tengah Utara", "E": "Belu", "T": "Belu", "F": "Sabu Raijua", "G": "Rote Ndao", "J": "Malaka"},
    "EB": {"default": "NTT (Flores)", "A": "Ende", "B": "Sikka", "C": "Flores Timur", "D": "Ngada", "E": "Manggarai", "F": "Lembata", "G": "Manggarai Barat", "H": "Nagekeo", "J": "Alor", "K": "Alor", "P": "Manggarai Timur"},
    "ED": {"default": "NTT (Sumba)", "A": "Sumba Timur", "B": "Sumba Barat", "C": "Sumba Barat Daya", "D": "Sumba Tengah"},
}

# -----------------------------------------------------------------------------
# COMMAND-LINE INTERFACE SETUP
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    # --- OS-Interchangeable Path Setup ---
    # Get the directory where this script is located.
    SCRIPT_DIR = Path(__file__).resolve().parent
    # Define the default model path relative to the script's directory.
    # This makes the script portable across Windows, macOS, and Linux.
    # **IMPORTANT**: Place your .pt model file in a 'models' subfolder.
    DEFAULT_MODEL_PATH = SCRIPT_DIR / "models" / "license_plate_detector.pt"

    parser = argparse.ArgumentParser(
        description="Real-time licence-plate detection, OCR, and categorization."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: 0 for webcam, path to video file, or RTSP/HTTP stream."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to the YOLO license plate detector model. Defaults to: {DEFAULT_MODEL_PATH}"
    )
    parser.add_argument(
        "--ocr-model",
        default="global-plates-mobile-vit-v2-model",
        help="Name of the fast-plate-ocr model to use."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on: 'cuda', 'cpu', or 'auto'."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Detection confidence threshold (e.g., 0.50 for 50%)."
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path to save the output video file (e.g., output.mp4)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the result window. (Default: True for webcam, False for files)."
    )
    return parser.parse_args()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# --- Constants for Drawing ---
TEXT_COLOR_BGR = (0, 255, 0)  # Green for text and boxes
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

def graceful_exit(cap=None, writer=None):
    """
    Releases all video resources and exits the script cleanly.
    This function is called on normal exit or via a signal (like Ctrl+C).
    """
    print("Exiting... releasing resources.")
    if cap:
        cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Finished. Bye 👋")
    sys.exit(0)

def get_plate_category(plate_txt: str) -> str:
    """
    Determines if a license plate is 'Ganjil' (odd) or 'Genap' (even)
    based on the last digit of its number.

    Args:
        plate_txt (str): The recognized license plate text.

    Returns:
        str: "Ganjil", "Genap", "No Number" if no digits are found, or "Invalid".
    """
    if not plate_txt:
        return ""

    # Corrected Regex: Find all digits (\d) in the string.
    digits = re.findall(r'\d', plate_txt)

    if not digits:
        return "No Number"

    # The category is determined by the very last digit found.
    try:
        last_digit = int(digits[-1])
        return "Genap" if last_digit % 2 == 0 else "Ganjil"
    except (ValueError, IndexError):
        return "Invalid"

def get_detailed_city_from_code(plate_txt: str, detailed_codes: dict) -> str:
    """
    Finds the specific city/regency from the license plate text by looking up
    its prefix and suffix in the provided dictionary.

    Args:
        plate_txt (str): The recognized license plate text.
        detailed_codes (dict): The dictionary mapping codes to regions.

    Returns:
        str: The name of the city/region, or "Unknown" if not found.
    """
    # 1. Sanitize the plate text to keep only uppercase letters and numbers.
    clean_plate = re.sub(r'[^A-Z0-9]', '', plate_txt.upper())
    if not clean_plate:
        return "Unknown"

    # 2. Identify the main region prefix (e.g., "AA", "B", "D").
    # We check for 2-letter prefixes first to avoid false matches (e.g., matching 'B' in 'BL').
    prefix = ""
    if len(clean_plate) >= 2 and clean_plate[:2] in detailed_codes:
        prefix = clean_plate[:2]
    elif clean_plate and clean_plate[0] in detailed_codes:
        prefix = clean_plate[0]

    if not prefix:
        return "Unknown"

    # 3. Find the first letter of the suffix (the first letter after the number block).
    # Corrected Regex: \d+ matches one or more digits.
    match = re.search(r'\d+([A-Z])', clean_plate)
    suffix_char = match.group(1) if match else ""

    # 4. Look up the region in the dictionary.
    region_data = detailed_codes.get(prefix, {})
    # Get the general region name as a fallback.
    general_region_name = region_data.get("default", f"Wilayah {prefix}")

    # If a suffix character was found, try to get the specific city.
    # Otherwise, or if the specific city isn't listed, return the general name.
    if suffix_char:
        return region_data.get(suffix_char, general_region_name)

    return general_region_name


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main() -> None:
    """
    The main function to run the license plate detection and OCR pipeline.
    """
    args = parse_args()

    # --- Step 0: Load Models ---
    print("Loading models...")
    try:
        detector = YOLO(args.model)
        ocr = ONNXPlateRecognizer(args.ocr_model, device=args.device)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load models. Ensure paths are correct and files are not corrupted. Details: {e}")
    print("Models loaded successfully.")

    # --- Step 1: Open Video Source ---
    # Convert source to int if it's a digit (for webcam index), otherwise use as is (for file path/URL).
    source_is_webcam = args.source.isdigit()
    src = int(args.source) if source_is_webcam else args.source
    
    print(f"Opening video source: {src}...")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video source: {args.source}")

    # --- Step 2: Setup Video Writer (if saving output) ---
    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Fallback to 30 FPS if not available
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Codec for .mp4 file
        writer = cv2.VideoWriter(str(args.save), fourcc, fps, (w, h))
        print(f"Saving output to: {args.save}")

    # --- Step 3: Setup Graceful Exit and Tracking ---
    # Register the graceful_exit function to be called on Ctrl+C (SIGINT).
    signal.signal(signal.SIGINT, lambda *_: graceful_exit(cap, writer))

    # This dictionary stores recently seen plates to avoid flooding the console with duplicate detections.
    # It maps a plate string to a deque (a fixed-size list) that acts as a simple flag.
    plate_memory: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=30))

    print("Starting video stream processing... Press 'q' in the window to quit.")
    ts_last = time.time() # For FPS calculation

    # --- Step 4: Main Processing Loop ---
    while True:
        ok, frame = cap.read()
        if not ok:
            print("End of video stream.")
            break

        # --- 4a: YOLO Detection ---
        # Perform license plate detection on the current frame.
        results = detector.predict(frame, conf=args.conf, verbose=False)
        boxes = results[0].boxes

        # --- 4b: Process Each Detected Plate ---
        for xyxy in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)

            # Crop the detected license plate from the frame.
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # --- 4c: OCR ---
            # Convert the crop to grayscale for better OCR performance.
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Run the OCR model on the grayscale crop.
            plate_txt = ocr.run(gray_crop)[0] if gray_crop.size else ""

            # --- 4d: Categorize and Get Region ---
            plate_category = get_plate_category(plate_txt)
            city = get_detailed_city_from_code(plate_txt, detailed_city_code_dict)

            # --- 4e: Draw Information on the Frame ---
            # Draw the bounding box around the plate.
            cv2.rectangle(frame, (x1, y1), (x2, y2), TEXT_COLOR_BGR, BOX_THICKNESS)

            # Prepare the labels to be displayed above the bounding box.
            labels = []
            if plate_txt:
                labels.append(f"Plat: {plate_txt}")
                if plate_category != "No Number":
                    labels.append(f"Tipe: {plate_category}")
                if city != "Unknown":
                    labels.append(f"Wilayah: {city}")
            else:
                labels.append("Membaca...")

            # Dynamically position the multi-line text overlay.
            font_scale = 0.6
            line_height = cv2.getTextSize("A", FONT, font_scale, 2)[0][1] + 10
            # Calculate the width of the widest label to create a fitting background.
            max_text_width = max((cv2.getTextSize(label, FONT, font_scale, 2)[0][0] for label in labels), default=0)

            # Calculate the top-left corner of the background rectangle.
            bg_y1 = y1 - (line_height * len(labels)) - 5
            # Draw the solid background rectangle.
            cv2.rectangle(frame, (x1, bg_y1), (x1 + max_text_width + 10, y1), TEXT_COLOR_BGR, -1)

            # Draw each label on a new line with a contrasting color (black).
            for i, label in enumerate(labels):
                text_y = y1 - (line_height * (len(labels) - 1 - i)) - 5
                cv2.putText(frame, label, (x1 + 5, text_y), FONT, font_scale, (0, 0, 0), 2, cv2.LINE_AA)

            # --- 4f: Log to Console (only for new plates) ---
            # Check if the plate is new by seeing if its 'memory' is all zeros (or empty).
            if plate_txt and sum(plate_memory[plate_txt]) == 0:
                plate_memory[plate_txt].appendleft(1) # Mark as seen.
                print(f"[{time.strftime('%H:%M:%S')}] Terdeteksi: {plate_txt} ({plate_category}, {city})")

        # --- 4g: Calculate and Display FPS ---
        now = time.time()
        fps = 1 / (now - ts_last)
        ts_last = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        # --- 4h: Show Frame and/or Write to File ---
        # Show window if --show is used OR if the source is a webcam.
        if args.show or source_is_webcam:
            cv2.imshow("Real-time Plate OCR (Refactored)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if writer:
            writer.write(frame)

    # --- Step 5: Cleanup ---
    graceful_exit(cap, writer)


if __name__ == "__main__":
    main()