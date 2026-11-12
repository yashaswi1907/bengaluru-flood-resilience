"""
Real-world reference points for Bengaluru used to ground the dataset.

Coordinates are approximate (geocoded to the locality / junction level, ~200-500 m accuracy).
Sources:
  - Flood hotspots: BBMP / KSNDMC vulnerable-location lists as reported by Deccan Herald,
    Citizen Matters, The News Minute, Public TV (2019-2025) and OpenCity
    "Flooding locations in Bengaluru Urban" dataset (KSRSAC / BBMP).
  - Lakes: BBMP / KSRSAC lake inventory (major lakes only).
  - Hospitals: major public & private hospitals (critical infrastructure).

Replace / extend these with the official KML/CSV files when you download them
(see src/real_data_loaders.py and DATA_SOURCES.md).
"""

# name, lat, lon, severity (2 = severely vulnerable, 1 = moderately vulnerable)
FLOOD_HOTSPOTS = [
    ("Silk Board Junction", 12.9172, 77.6229, 2),
    ("Bellandur / Ecospace ORR", 12.9260, 77.6760, 2),
    ("Sai Layout, Horamavu", 13.0330, 77.6600, 2),
    ("Rainbow Drive Layout, Sarjapur Rd", 12.9100, 77.6830, 2),
    ("HRBR Layout", 13.0220, 77.6450, 2),
    ("Veerannapalya / Manyata", 13.0450, 77.6200, 2),
    ("Rajajinagar Industrial Area", 12.9900, 77.5500, 2),
    ("Nayandahalli Junction", 12.9420, 77.5230, 2),
    ("Gali Anjaneya Temple Junction", 12.9450, 77.5350, 2),
    ("Bannerghatta Road (Hulimavu)", 12.8780, 77.6050, 2),
    ("Jayadeva Flyover", 12.9180, 77.6000, 1),
    ("Koramangala (ST Bed / 6th Block)", 12.9350, 77.6200, 2),
    ("Ejipura", 12.9420, 77.6300, 1),
    ("Shanthinagar / Double Road", 12.9560, 77.5980, 1),
    ("Okalipuram Underpass", 12.9800, 77.5620, 1),
    ("Kino Theatre Underpass, Sheshadripuram", 12.9900, 77.5720, 1),
    ("Shivananda Circle", 12.9880, 77.5850, 1),
    ("Jayamahal Road", 13.0000, 77.6000, 1),
    ("Indiranagar 80 ft Road", 12.9700, 77.6450, 1),
    ("Kalyan Nagar 100 ft Road", 13.0250, 77.6400, 1),
    ("Kadugodi", 12.9980, 77.7600, 1),
    ("KR Puram / Hoodi", 13.0000, 77.6950, 2),
    ("Mahadevapura", 12.9900, 77.6900, 2),
    ("Munnekolala", 12.9550, 77.7150, 1),
    ("Varthur", 12.9400, 77.7450, 2),
    ("Hebbal Flyover / Kempapura", 13.0400, 77.5950, 1),
    ("Kendriya Vihar, Yelahanka", 13.0900, 77.5800, 2),
    ("Kogilu Cross", 13.1000, 77.6100, 1),
    ("Vidyaranyapura", 13.0800, 77.5600, 1),
    ("Doddabidarakallu", 13.0400, 77.4800, 1),
    ("Mysuru Road (Kengeri)", 12.9100, 77.4850, 1),
    ("RR Nagar (Ideal Homes)", 12.9250, 77.5150, 1),
    ("Bommanahalli", 12.9000, 77.6300, 2),
    ("HSR Layout Sector 6", 12.9100, 77.6450, 1),
    ("Madiwala", 12.9200, 77.6150, 1),
    ("Arekere / Anugraha Layout", 12.8850, 77.6000, 1),
    ("Yelachenahalli / Kanakapura Rd", 12.8950, 77.5700, 1),
    ("Hennur", 13.0350, 77.6400, 1),
    ("Panathur Underpass", 12.9350, 77.7000, 1),
    ("Mahadevapura Outer Ring Rd (Doddanekundi)", 12.9750, 77.6950, 1),
]

# name, lat, lon, approx area (ha)
LAKES = [
    ("Bellandur Lake", 12.9350, 77.6650, 360),
    ("Varthur Lake", 12.9450, 77.7400, 180),
    ("Hebbal Lake", 13.0450, 77.5900, 60),
    ("Ulsoor Lake", 12.9820, 77.6200, 50),
    ("Madiwala Lake", 12.9080, 77.6170, 110),
    ("Agara Lake", 12.9220, 77.6420, 40),
    ("Sankey Tank", 13.0100, 77.5750, 15),
    ("Yelahanka Lake", 13.0950, 77.5900, 120),
    ("Puttenahalli Lake (Yelahanka)", 13.0800, 77.5850, 50),
    ("Lalbagh Lake", 12.9490, 77.5860, 12),
    ("Kaikondrahalli Lake", 12.9120, 77.6780, 20),
    ("Hulimavu Lake", 12.8770, 77.6100, 55),
    ("Jakkur Lake", 13.0750, 77.6050, 65),
    ("Rachenahalli Lake", 13.0560, 77.6250, 50),
    ("KR Puram (Kalkere) Lake", 13.0200, 77.6800, 45),
    ("Nagavara Lake", 13.0400, 77.6100, 25),
    ("Kengeri Lake", 12.9150, 77.4900, 30),
    ("Kommaghatta / Ullal Lake", 12.9500, 77.5000, 35),
    ("Puttenahalli Lake (JP Nagar)", 12.8920, 77.5850, 5),
    ("Doddanekundi Lake", 12.9780, 77.6950, 45),
    ("Sheelavanthana Kere / Kundalahalli", 12.9600, 77.7150, 15),
    ("Allalasandra Lake", 13.0850, 77.5700, 20),
    ("Doddabidarakallu Lake", 13.0400, 77.4850, 20),
    ("Horamavu Agara Lake", 13.0300, 77.6650, 35),
]

# name, lat, lon, beds (approx)
HOSPITALS = [
    ("Victoria Hospital", 12.9600, 77.5740, 1000),
    ("Bowring & Lady Curzon Hospital", 12.9830, 77.6040, 700),
    ("NIMHANS", 12.9420, 77.5960, 900),
    ("St. John's Medical College Hospital", 12.9310, 77.6200, 1300),
    ("Manipal Hospital Old Airport Rd", 12.9590, 77.6480, 600),
    ("Jayadeva Institute of Cardiology", 12.9170, 77.5990, 1000),
    ("KC General Hospital", 12.9930, 77.5670, 400),
    ("Narayana Health City", 12.8100, 77.6950, 1400),
    ("Columbia Asia Hebbal", 13.0480, 77.5920, 150),
    ("Sakra World Hospital", 12.9310, 77.6830, 300),
    ("Vydehi Hospital, Whitefield", 12.9770, 77.7280, 1600),
    ("MS Ramaiah Hospital", 13.0300, 77.5680, 1200),
    ("Fortis Bannerghatta Rd", 12.8950, 77.5990, 400),
    ("BGS Gleneagles, Kengeri", 12.9050, 77.4990, 500),
    ("Aster CMI, Hebbal", 13.0490, 77.5930, 500),
    ("Sanjay Gandhi Trauma Centre", 12.9200, 77.5970, 200),
    ("Yelahanka General Hospital", 13.1000, 77.5960, 100),
    ("KR Puram General Hospital", 13.0050, 77.6950, 100),
]

# BBMP-era administrative zones (approximate centroids) for reporting
ZONES = [
    ("East", 12.985, 77.620),
    ("West", 12.985, 77.555),
    ("South", 12.930, 77.580),
    ("Bommanahalli", 12.895, 77.630),
    ("Mahadevapura", 12.980, 77.715),
    ("Yelahanka", 13.090, 77.590),
    ("Dasarahalli", 13.045, 77.510),
    ("RR Nagar", 12.925, 77.510),
]

CITY_CENTER = (12.9716, 77.5946)
