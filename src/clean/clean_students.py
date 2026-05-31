# =============================================================
# clean_students.py
# =============================================================
# Clean the raw students export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why this is by far the most complex cleaning script:
#   Student records from Registan's system are entered manually by
#   reception staff — often under time pressure — and contain a wide
#   variety of data quality problems that I discovered by inspecting
#   the raw JSON:
#
#   1. Bb prefix in first names: Some staff prepend "Bb " or "B " to
#      names (possibly a CRM shorthand). I strip this prefix.
#
#   2. Cyrillic names: About 30% of names are in Cyrillic script.
#      I transliterate them to Latin so all names are in a consistent
#      alphabet. This matters because the dashboard displays names and
#      gender is detected from name dictionaries that are Latin-only.
#
#   3. Emojis: Some records contain emojis in name fields. I remove
#      them because they cause encoding errors in downstream CSV exports.
#
#   4. Phone numbers in last name field: Some staff entered phone
#      numbers in the lastName column when the phoneNumber field was
#      empty. I rescue these numbers and move them to the correct field.
#
#   5. Name swaps: Some records have the surname in firstName and the
#      given name in lastName. I detect this using Uzbek surname suffixes
#      (ov, ova, ev, eva etc.) and swap the fields when needed.
#
#   6. Gender detection: Registan does not collect gender at registration.
#      I infer it from the first name using a dictionary of 80+ female and
#      80+ male Uzbek names, backed up by suffix rules (ova/eva = female,
#      ov/ev = male). My supervisor asked me to include gender because it
#      may interact with dropout patterns (e.g. female students in evening
#      shifts may face different constraints than male students).
#
#   7. Bulk import flag: Records created before 2022-05-01 were bulk-
#      imported from a paper registry — their timestamps are unreliable.
#      I flag these rows so build_master.py can exclude them from any
#      time-based analysis.
#
# Why I filter to Chilonzor branch during streaming:
#   The students JSON contains students from all branches. I filter to
#   Chilonzor during the streaming loop (not after building the dataframe)
#   to avoid loading ~15,000 irrelevant students into memory.
#
# Input : data/raw/students.raw.json
# Output: data/interim/students.parquet
# =============================================================

import ijson
import pandas as pd
import re
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import STUDENTS as RAW, STUDENTS_CLEAN as OUT, CHILONZOR_BRANCH_ID as BRANCH_ID

def _oid(val):
    """Handle both MongoDB {'$oid': '...'} and plain string IDs from the API."""
    if isinstance(val, dict):
        return val.get("$oid")
    return val

# ── cyrillic → latin map ───────────────────────────────────────
# I transliterate rather than drop Cyrillic names because dropping
# would delete ~30% of student records. The mapping follows the
# standard Uzbek Latin alphabet where possible.
CYRILLIC_MAP = {
    'А':'A','а':'a','Б':'B','б':'b','В':'V','в':'v',
    'Г':'G','г':'g','Д':'D','д':'d','Е':'Ye','е':'ye',
    'Ё':'Yo','ё':'yo','Ж':'J','ж':'j','З':'Z','з':'z',
    'И':'I','и':'i','Й':'Y','й':'y','К':'K','к':'k',
    'Л':'L','л':'l','М':'M','м':'m','Н':'N','н':'n',
    'О':'O','о':'o','П':'P','п':'p','Р':'R','р':'r',
    'С':'S','с':'s','Т':'T','т':'t','У':'U','у':'u',
    'Ф':'F','ф':'f','Х':'X','х':'x','Ц':'Ts','ц':'ts',
    'Ч':'Ch','ч':'ch','Ш':'Sh','ш':'sh','Щ':'Sh','щ':'sh',
    'Ъ':"'",'ъ':"'",'Ы':'I','ы':'i','Ь':"'",'ь':"'",
    'Э':'E','э':'e','Ю':'Yu','ю':'yu','Я':'Ya','я':'ya',
    'Ў':"O'",'ў':"o'",'Қ':'Q','қ':'q',
    'Ғ':"G'",'ғ':"g'",'Ҳ':'H','ҳ':'h',
}

def transliterate(text):
    if not text:
        return ""
    return ''.join(CYRILLIC_MAP.get(c, c) for c in text)

# ── emoji removal ──────────────────────────────────────────────
# Emojis cause encoding errors in downstream CSV exports. I strip
# them by encoding to ASCII (which drops non-ASCII characters) and
# then cleaning up any residual non-word characters.
def remove_emojis(text):
    if not text:
        return ""
    cleaned = text.encode('ascii', 'ignore').decode('ascii')
    cleaned = re.sub(r'[^\w\s\'\-\.\,\/]', '', cleaned)
    return cleaned.strip()

# ── chinese surname detection ──────────────────────────────────
# Registan has a small number of Chinese students. Their names follow
# a different structure (surname first) so I detect them and keep two
# words (surname + given name) rather than truncating to one word.
CHINESE_SURNAMES = {
    'wang','zhang','li','liu','chen','luo','hao',
    'zhao','wu','zhou','xu','sun','ma','zhu','hu',
    'guo','he','lin','gao','liang','zheng','xiao'
}

def is_chinese_name(word):
    return word.lower() in CHINESE_SURNAMES

# ── bb prefix cleaning ─────────────────────────────────────────
# The "Bb " prefix appears in roughly 200 records. I believe it was
# entered by a specific receptionist as a personal shorthand.
# I strip it before any other processing.
def clean_first_name(raw):
    if not raw:
        return ""
    text = raw.strip()
    if re.match(r'^[Bb][Bb]?\s+', text):
        text = re.sub(r'^[Bb][Bb]?\s+', '', text).strip()
    words = text.split()
    if not words:
        return ""
    first_word = words[0]
    if is_chinese_name(first_word) and len(words) >= 2:
        return f"{words[0]} {words[1]}".title()
    return first_word.title()

# ── last name cleaning + phone rescue ─────────────────────────
# The phone rescue logic tries three common Uzbek phone formats.
# If the student has no phone number but their last name contains
# a pattern that looks like a phone number, I extract it and move
# it to the phoneNumber field.
PHONE_PATTERNS = [
    r'9\d{8}',
    r'\d{2}-\d{3}-\d{2}-\d{2}',
    r'\d{9,12}',
]

def clean_last_name(raw, current_phone):
    if not raw:
        return "", None
    text = str(raw).strip()
    rescued_phone = None

    if current_phone is None:
        for pattern in PHONE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                phone_candidate = re.sub(r'[^\d]', '', match.group())
                if len(phone_candidate) == 9:
                    rescued_phone = '+998' + phone_candidate
                elif len(phone_candidate) == 12 and phone_candidate.startswith('998'):
                    rescued_phone = '+' + phone_candidate
                break

    # strip parenthetical content, phone patterns, numbers, and
    # Registan-specific tag words that staff sometimes add to names
    text = re.sub(r'\(.*?\)', '', text)
    for pattern in PHONE_PATTERNS:
        text = re.sub(pattern, '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[-/\\]', ' ', text)
    text = remove_emojis(text)
    text = re.sub(
        r'\b(maktab|grant|kids|ya|yash|ustoz|support|'
        r'sanada|sanalarda|majburiy|dan|kuni)\b',
        '', text, flags=re.IGNORECASE
    )
    text = ' '.join(text.split()).title().strip()
    if not text or text in ['"', "'", '000', '""']:
        return "", rescued_phone
    return text, rescued_phone

# ── surname suffix detection ───────────────────────────────────
# Uzbek surnames follow predictable suffix patterns. I use these
# to detect when first and last names have been swapped by the
# staff member entering the data.
SURNAME_SUFFIXES = (
    'ov','ev','ova','eva','yev','yeva','off','eff',
    'qizi','ugli','oglu','bekov','bekova'
)

def looks_like_surname(name):
    if not name:
        return False
    return any(name.lower().endswith(s) for s in SURNAME_SUFFIXES)

# ── gender dictionary ──────────────────────────────────────────
# I built these lists by inspecting the most common names in the
# student data and cross-referencing with Uzbek name databases.
# The suffix fallback (layer 2) handles names not in either list.
FEMALE_NAMES = {
    'madina','sevinch','marjona','aziza','muslima','maftuna',
    'sarvinoz','farangiz','durdona','iroda','diyora','nilufar',
    'feruza','robiya','shahzoda','munisa','nozima','sevara',
    'umida','dilnoza','mohinur','muxlisa','shahnoza','dildora',
    'jasmina','mubina','ruxshona','zarina','charos','sabina',
    'malika','hulkar','kamola','barno','nasiba','zulfiya',
    'oydin','oydinoy','oysha','oynur','shahlo','sitora',
    'nargiza','dilorom','gulnora','mavluda','nafosa','hamida',
    'farzona','fotima','xurmo','mohira','lola','nodira',
    'zilola','yulduz','manzura','adolat','zuhra','lobar',
    'latofat','muazzam','xilola','gulsanam','irodaxon',
    'sadoqat','rayxona','sabohat','dilfuza','mahliyo','ziyoda',
    'kumush','laylo','sitorabonu','mehribon','mushtariy',
    'shaxlo','tabassum','gavharoy','munavvar','shaxzoda',
    'nafisa','dilnavoz','risolatxon','nozimaxon','gulbahor',
    'shahnozaxon','dilrabo','muxabbat','rohila','sarvinozxon',
    'gulsara','muazzamxon','gulasal','intizor','mohiniso',
    'begoyim','mumtoza','dilsora','navbaxor','rayhona',
    'shodiyona','xonzoda','rano','zebo','ifora','shodiya',
    'orasta','guldona','shahribonu','asemay','oynisa',
    "o'g'iloy",'xurliman','mashhura','robiyabonu','madinabonu',
    'gulhayo','mehriniso','xurshida','kumushbibi','dilyara',
    'gulibonu','muqaddas','jannur','davlatbeka','marhamat',
    'ergashoy','nigina','taxmina','shirin','amina','ra\'no',
    'ozodabonus','tabassum',
}

MALE_NAMES = {
    'sardor','javohir','azizbek','behruz','asadbek','jahongir',
    'abdulaziz','diyorbek','asilbek','samandar','otabek',
    'dilshod','bobur','jamshid','muhammad','ibrohim','ulugbek',
    'abdulloh','ozodbek','jasur','mirzo','bekzod','nodir',
    'javlon','temur','doniyor','firdavs','eldor','sanjar',
    'sherzod','ismoil','islom','husan','husniddin','muzaffar',
    'mansur','murod','bahrom','baxtiyor','alisher','anvar',
    'rustam','timur','akbar','akmal','ali','aziz','bahodir',
    'davron','farrux','hamza','hamid','hasan','ilhom',
    'kamol','komil','nurbek','odil','ortiq','oybek',
    'parviz','qodir','ravshan','saidakbar','sarvar',
    'shamsiddin','shuhrat','sirojiddin','suxrob','tohir',
    'ulmas','umid','uygun','vohid','xurshid','yoqubjon',
    'zafar','zohid','zubaydullo','shohjahon','shohrux',
    'abdumalik','abdurahmon','abdunabi','abdujalil',
    'abduazim','abdullatif','abdulbosit','ahmadxoja','axror',
    'azimjon','bilol','boburjon','burxon','davronbek',
    'dostonbek','elshod','elyor','elbek','erxan','fayozbek',
    'halilbek','hikmatillo','hojiakbar','ibrat','ikrom',
    'ilyos','ilyosiddin','iskandar','izzatilla','jalolbek',
    'jasurbek',"jo'rabek",'kamronbek','kozimjon','lutfullo',
    'mahmudjon','mansurxon','maqsadbek','mirafzal','mirkomil',
    'mirzohid','muhammadali','muhammadjon','muhammadsobir',
    'muhammadsolih','muhammadyusuf','muhsinbek','munis',
    'murodjon','navruzbek',"ne'matullo",'norillo','nortoji',
    'nodirbek','nurmuhammad','nurulloh','odilbek','odilxon',
    'olimjon','omadbek','quvonchbek','rasul','saidazim',
    'sanjarbek','sayfullo','sherozbek','shoxrux','sirojbek',
    'sodiqjon','suhrob','sulaymon','sunnatilla','toxirjon',
    'ubaydullo',"ulug'bek",'umarali','usmonjon','xojiakbar',
    'xursanmurod','yahyobek','zafar','zikrullo','zubaydulloh',
    'dzakhon','arman','akobir','ahadbek','asrorbek','azimxon',
    'elyorbek','mironshoh',"mirsaidxo'ja",'nosir','bexruz',
    'baxtinur','mexriddin','faxriddin','shahobiddin','kamoliddin',
    'jaloliddin','asliddin','sirojiddin','shamsiddin',
    'hasanboy','tursunboy','karimboy','odamboy','xasanboy',
    'roziboy','olimboy','talanboy','otaboy','donoboy',
    'meliboy','esomboy','mehriboy','allayorboy','aminboy',
    'bozorboy','jumanboy','mavsumboy','omonboy','salimboy',
    'hayitboy','erkaboy','umonboy','maktaboy','riqsiboy',
    "ro'ziboy","to'xtaboy","o'rolboy",
}

def detect_gender(first_name, language, mode_gender):
    if not first_name:
        return mode_gender
    name_clean = first_name.lower().strip()

    # layer 1 — dictionary lookup (most reliable)
    if name_clean in FEMALE_NAMES:
        return 'female'
    if name_clean in MALE_NAMES:
        return 'male'

    # layer 2 — suffix rules (catches names not in dictionary)
    female_suffixes = ('ova','eva','yeva','ina','ara','iya',
                       'ira','oxon','xon','gul','noz','ona',
                       'oy','oyi','bonu','roy','zod')
    male_suffixes   = ('ov','ev','yev','bek','jon','din',
                       'boy','bay','mir','off','ugli','oglu')
    if any(name_clean.endswith(s) for s in female_suffixes):
        return 'female'
    if any(name_clean.endswith(s) for s in male_suffixes):
        return 'male'

    # layer 3 — fall back to the mode gender for this language group
    return mode_gender

# ── phone normalization ────────────────────────────────────────
# I normalise all phone numbers to the +998XXXXXXXXX format.
# Uzbekistan phone numbers are 9 digits after the country code.
def normalize_phone(val):
    if not val:
        return None, True
    val = re.sub(r'[\s\-\(\)]', '', str(val))
    if re.match(r'^\+998\d{9}$', val):
        return val, False
    if re.match(r'^998\d{9}$', val):
        return '+' + val, False
    if re.match(r'^\d{9}$', val):
        return '+998' + val, False
    return None, True

# ── step 1: stream and extract ─────────────────────────────────
# I filter to Chilonzor branch during streaming (not after building
# the dataframe) to avoid loading irrelevant students into memory.
# Students without a Chilonzor branch entry (state is None) are
# from other branches and are skipped entirely.
if not RAW.exists():
    print(f"⚠  {RAW.name} not found (no new students since last fetch — API returned 0).")
    print("   Saving empty students.parquet so downstream scripts don't crash.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["studentId","fullName","firstName","phoneNumber",
                           "state","isDeleted","createdAt","updatedAt",
                           "archiveDate","graduatedAt","moderatorId"]).to_parquet(OUT, index=False)
    print(f"✅ Empty students saved → {OUT}")
    import sys; sys.exit(0)

print("Reading students.raw.json ...")
records = []

with open(RAW, "rb") as f:
    for student in ijson.items(f, "item"):

        sid         = _oid(student.get("_id"))
        raw_first   = student.get("firstName") or ""
        raw_last    = student.get("lastName") or ""
        full_name   = student.get("fullName") or ""
        phone_raw   = student.get("phoneNumber") or ""
        language    = student.get("language")
        created_at  = student.get("createdAt")
        deleted_at  = student.get("deletedAt", 0)
        coins       = student.get("coinsPoint") or 0
        has_email   = bool(student.get("email"))
        is_referred = bool(student.get("referredByStudentId"))
        groups_count    = student.get("groupsCount") or 0
        referrals_count = student.get("referralsCount") or 0
        category_id = _oid(student.get("categoryId"))
        source_id   = _oid(student.get("sourceId"))

        state             = None
        balance           = 0
        moderator_id      = None
        archive_date      = None
        graduated_at      = None
        state_active_date = None
        state_new_date    = None
        order_created     = None
        order_cancelled   = None

        for branch in student.get("branches", []):
            bid = _oid(branch.get("branchId")) or ""
            if bid == BRANCH_ID:
                state             = branch.get("state")
                balance           = branch.get("balance") or 0
                moderator_id      = _oid(branch.get("moderatorId"))
                archive_date      = branch.get("archiveDate")
                graduated_at      = branch.get("graduatedAt")
                state_active_date = branch.get("stateActiveDate")
                state_new_date    = branch.get("stateNewDate")
                order_created     = branch.get("orderCreatedDate")
                order_cancelled   = branch.get("orderCancelledDate")
                break

        if state is None:
            continue

        phone, phone_invalid = normalize_phone(phone_raw)

        records.append({
            "studentId"          : sid,
            "rawFirstName"       : raw_first,
            "rawLastName"        : raw_last,
            "fullName"           : full_name,
            "phoneNumber"        : phone,
            "phoneInvalid"       : phone_invalid,
            "language"           : language,
            "createdAt"          : created_at,
            "isDeleted"          : deleted_at != 0,
            "coinsPoint"         : coins,
            "groupsCount"        : groups_count,
            "referralsCount"     : referrals_count,
            "hasEmail"           : has_email,
            "isReferred"         : is_referred,
            "categoryId"         : category_id,
            "sourceId"           : source_id,
            "state"              : state,
            "balance"            : balance,
            "moderatorId"        : moderator_id,
            "archiveDate"        : archive_date,
            "graduatedAt"        : graduated_at,
            "stateActiveDate"    : state_active_date,
            "stateNewDate"       : state_new_date,
            "orderCreatedDate"   : order_created,
            "orderCancelledDate" : order_cancelled,
        })

print(f"  → {len(records)} students found in Chilonzor branch")

# ── step 2: dataframe + types ──────────────────────────────────
df = pd.DataFrame(records)

date_cols = ["createdAt","archiveDate","graduatedAt",
             "stateActiveDate","stateNewDate",
             "orderCreatedDate","orderCancelledDate"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

df["balance"]        = pd.to_numeric(df["balance"],        errors="coerce").fillna(0).astype(int)
df["coinsPoint"]     = pd.to_numeric(df["coinsPoint"],     errors="coerce").fillna(0).astype(int)
df["groupsCount"]    = pd.to_numeric(df["groupsCount"],    errors="coerce").fillna(0).astype(int)
df["referralsCount"] = pd.to_numeric(df["referralsCount"], errors="coerce").fillna(0).astype(int)
# fill missing language with the mode — very few nulls, safe imputation
df["language"]       = df["language"].fillna(df["language"].mode()[0])

# ── step 3: clean names ────────────────────────────────────────
print("Cleaning names ...")

ALL_GIVEN = FEMALE_NAMES | MALE_NAMES

first_names    = []
last_names     = []
rescued_phones = []

for _, row in df.iterrows():
    fn = clean_first_name(row["rawFirstName"])
    fn = remove_emojis(transliterate(fn))

    current_phone = row["phoneNumber"]
    ln, rescued   = clean_last_name(row["rawLastName"], current_phone)
    ln = remove_emojis(transliterate(ln))

    # case 1: firstName looks like a surname AND lastName is empty → move to lastName
    if looks_like_surname(fn) and ln.strip() == "":
        ln = fn
        fn = ""
    # case 2: firstName looks like a surname AND lastName looks like a given name → swap
    elif looks_like_surname(fn) and not looks_like_surname(ln) and len(ln) > 3 and ln.replace("'","").replace(" ","").isalpha():
        fn, ln = ln, fn

    first_names.append(fn.title().strip())
    last_names.append(ln.title().strip())
    rescued_phones.append(rescued)

df["firstName"] = first_names
df["lastName"]  = last_names
# Rebuild fullName from cleaned first+last so it is always emoji-free and transliterated
df["fullName"]  = (df["firstName"] + " " + df["lastName"]).str.strip()

# apply rescued phones where the student previously had no number
for i, rescued in enumerate(rescued_phones):
    if rescued and df.at[i, "phoneNumber"] is None:
        phone, invalid = normalize_phone(rescued)
        df.at[i, "phoneNumber"]  = phone
        df.at[i, "phoneInvalid"] = invalid

df = df.drop(columns=["rawFirstName", "rawLastName"])

# ── step 4: gender ─────────────────────────────────────────────
# Three-layer detection: dictionary → suffix rules → language-group mode.
# The mode fallback uses the most common detected gender for each
# language group (Uzbek / Russian / English students) rather than a
# global mode, because gender ratios differ across language groups.
print("Detecting gender ...")

temp_g = df["firstName"].str.lower().map(
    lambda n: 'female' if n in FEMALE_NAMES else ('male' if n in MALE_NAMES else None)
)
df["_tg"] = temp_g
mode_gender_by_lang = df.groupby("language")["_tg"].agg(
    lambda x: x.dropna().mode()[0] if x.dropna().shape[0] > 0 else 'female'
)

df["gender"] = df.apply(
    lambda row: detect_gender(
        row["firstName"],
        row["language"],
        mode_gender_by_lang.get(row["language"], "female")
    ), axis=1
)
df = df.drop(columns=["_tg"])

# ── step 5: time features ──────────────────────────────────────
# I derive joinMonth, joinYear, joinSeason, and academicYear from
# createdAt. These are used in build_master.py as demographic features.
# joinSeason captures whether a student started in a new-academic-year
# rush (September) or a summer enrolment window — both are known to
# have different dropout dynamics in educational settings.
print("Computing time features ...")

df["joinMonth"]  = df["createdAt"].dt.to_period("M").astype(str)
df["joinYear"]   = df["createdAt"].dt.year
df["joinSeason"] = df["createdAt"].dt.month.map({
    12:"winter",1:"winter",2:"winter",
    3:"spring", 4:"spring", 5:"spring",
    6:"summer", 7:"summer", 8:"summer",
    9:"autumn", 10:"autumn",11:"autumn"
})

def academic_year(dt):
    if pd.isna(dt):
        return None
    y, m = dt.year, dt.month
    return f"{y}-{y+1}" if m >= 9 else f"{y-1}-{y}"

df["academicYear"] = df["createdAt"].apply(academic_year)

# ── step 6: bulk import flag ───────────────────────────────────
# Records created before 2022-05-01 were bulk-imported from paper
# records rather than entered in real time. Their timestamps are
# unreliable — many share the same createdAt date, which is actually
# the date of the import operation, not the date of enrolment.
# I flag these rows so downstream scripts can exclude them from
# any time-series analysis that depends on accurate join dates.
bulk_cutoff        = pd.Timestamp("2022-05-01", tz="UTC")
df["isBulkImport"] = df["createdAt"] < bulk_cutoff

# ── step 7: summary ────────────────────────────────────────────
print("\n── State distribution ──")
print(df["state"].value_counts())
print("\n── Gender distribution ──")
print(df["gender"].value_counts())
print("\n── Language distribution ──")
print(df["language"].value_counts())
print("\n── Bulk import ──")
print(df["isBulkImport"].value_counts())
print("\n── Phone issues ──")
print(f"  Invalid/missing : {df['phoneInvalid'].sum()}")
print(f"  No moderator    : {df['moderatorId'].isna().sum()}")
print("\n── Students per academic year ──")
print(df.groupby("academicYear").size().to_string())
print("\n── Sample cleaned names ──")
print(df[["firstName","lastName","gender"]].head(20).to_string())
print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")
print(f"Columns       : {list(df.columns)}")

# ── step 8: save ───────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
