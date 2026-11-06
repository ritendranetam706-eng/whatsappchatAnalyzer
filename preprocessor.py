"""
preprocessor.py
================
Robust WhatsApp chat export preprocessing engine.
Handles standard 12-hour (AM/PM) and 24-hour formats across Android and iOS exports.
Performs regex extraction, message parsing, datetime feature engineering,
and categorisation of system notifications and media placeholders.
"""

import re
import pandas as pd


# Regex pattern to identify media omission placeholders across platforms
MEDIA_PATTERN = re.compile(
    r'(<Media omitted>|'
    r'<image omitted>|<video omitted>|<audio omitted>|<document omitted>|<sticker omitted>|'
    r'image omitted|video omitted|audio omitted|document omitted|sticker omitted|'
    r'Contact card omitted|\u200e?attached:)',
    re.IGNORECASE
)

# Regex pattern to identify deleted message indicators
DELETED_PATTERN = re.compile(
    r'(This message was deleted|You deleted this message)',
    re.IGNORECASE
)

# Common WhatsApp system notification indicators
SYSTEM_INDICATORS = [
    "messages and calls are end-to-end encrypted",
    "created group",
    "added",
    "removed",
    "left",
    "changed the group description",
    "changed the subject",
    "changed this group's icon",
    "changed the group icon",
    "security code changed",
    "you're now an admin",
    "joined using this group's invite link",
    "waiting for this message. this may take a while."
]


def _clean_unicode_artifacts(text: str) -> str:
    """Strip invisible directional markers and normalise non-breaking spaces."""
    return (
        text.replace('\u202f', ' ')
            .replace('\u00a0', ' ')
            .replace('\u200e', '')
            .replace('\u200f', '')
    )


def preprocess(data: str) -> pd.DataFrame:
    """
    Parses raw WhatsApp chat export text into a structured, feature-rich Pandas DataFrame.

    Parameters:
        data (str): Raw string content of WhatsApp exported .txt file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with columns:
            ['date', 'only_date', 'year', 'month_num', 'month', 'day',
             'day_name', 'hour', 'minute', 'period', 'user', 'message',
             'is_media', 'is_deleted', 'is_system']

    Raises:
        ValueError: If no valid timestamp pattern is detected or file content is empty.
    """
    if not data or not data.strip():
        raise ValueError("The uploaded chat file is empty.")

    # Normalise whitespace / unicode markers
    data = _clean_unicode_artifacts(data)

    # -------------------------------------------------------------------------
    # Detection Profiles: Android & iOS formats (12-hour and 24-hour)
    # -------------------------------------------------------------------------
    profiles = [
        # Android 12-hour (e.g., "12/05/2023, 10:45 PM - ")
        {
            "pattern": r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)\s-\s)',
            "is_bracket": False,
            "has_seconds": False,
            "has_ampm": True
        },
        # Android 24-hour (e.g., "12/05/2023, 22:45 - ")
        {
            "pattern": r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s)',
            "is_bracket": False,
            "has_seconds": False,
            "has_ampm": False
        },
        # iOS 12-hour with seconds (e.g., "[12/05/2023, 10:45:12 PM] ")
        {
            "pattern": r'(\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s?(?:AM|PM|am|pm)\]\s)',
            "is_bracket": True,
            "has_seconds": True,
            "has_ampm": True
        },
        # iOS 12-hour without seconds (e.g., "[12/05/2023, 10:45 PM] ")
        {
            "pattern": r'(\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)\]\s)',
            "is_bracket": True,
            "has_seconds": False,
            "has_ampm": True
        },
        # iOS 24-hour with seconds (e.g., "[12/05/2023, 22:45:12] ")
        {
            "pattern": r'(\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\]\s)',
            "is_bracket": True,
            "has_seconds": True,
            "has_ampm": False
        },
        # iOS 24-hour without seconds (e.g., "[12/05/2023, 22:45] ")
        {
            "pattern": r'(\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\]\s)',
            "is_bracket": True,
            "has_seconds": False,
            "has_ampm": False
        },
    ]

    matched_profile = None
    for p in profiles:
        if re.search(p["pattern"], data):
            matched_profile = p
            break

    # Fallback generic pattern if specific profile did not match
    if matched_profile is None:
        generic_pattern = r'(\[?\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\]?\s?(?:-\s)?)'
        if re.search(generic_pattern, data):
            matched_profile = {
                "pattern": generic_pattern,
                "is_bracket": False,
                "has_seconds": False,
                "has_ampm": False
            }
        else:
            raise ValueError(
                "Unrecognised WhatsApp export timestamp format. "
                "Ensure you export the chat as a plain .txt file without media."
            )

    split_pattern = matched_profile["pattern"]

    # Split chat into timestamp markers and message contents
    tokens = re.split(split_pattern, data)

    # If first token does not match timestamp, it's leading metadata or empty string
    raw_messages = []
    raw_dates = []

    # re.split with capturing group gives: [leading_text, match1, text1, match2, text2, ...]
    if len(tokens) >= 3:
        for i in range(1, len(tokens), 2):
            raw_dates.append(tokens[i])
            raw_messages.append(tokens[i + 1] if i + 1 < len(tokens) else "")
    else:
        # Fallback using findall and split
        raw_dates = re.findall(split_pattern, data)
        raw_messages = re.split(split_pattern, data)[1:]

    if not raw_dates or not raw_messages:
        raise ValueError("Could not extract any valid messages from the chat export.")

    # Clean date strings
    def clean_timestamp(ts: str) -> str:
        s = ts.strip()
        s = s.strip('[]')
        s = re.sub(r'\s?-\s?$', '', s)
        return s.strip()

    cleaned_dates = [clean_timestamp(d) for d in raw_dates]

    # Align lengths if uneven
    min_len = min(len(cleaned_dates), len(raw_messages))
    cleaned_dates = cleaned_dates[:min_len]
    raw_messages = raw_messages[:min_len]

    # Construct initial DataFrame
    df = pd.DataFrame({
        'user_message': raw_messages,
        'message_date': cleaned_dates
    })

    # -------------------------------------------------------------------------
    # Datetime Parsing with Multiple Fallbacks
    # -------------------------------------------------------------------------
    date_formats_to_try = [
        # 12-hour formats
        '%d/%m/%Y, %I:%M %p',
        '%d/%m/%y, %I:%M %p',
        '%m/%d/%Y, %I:%M %p',
        '%m/%d/%y, %I:%M %p',
        '%d/%m/%Y, %I:%M:%S %p',
        '%d/%m/%y, %I:%M:%S %p',
        '%m/%d/%Y, %I:%M:%S %p',
        '%m/%d/%y, %I:%M:%S %p',
        # 24-hour formats
        '%d/%m/%Y, %H:%M',
        '%d/%m/%y, %H:%M',
        '%m/%d/%Y, %H:%M',
        '%m/%d/%y, %H:%M',
        '%d/%m/%Y, %H:%M:%S',
        '%d/%m/%y, %H:%M:%S',
        '%m/%d/%Y, %H:%M:%S',
        '%m/%d/%y, %H:%M:%S',
    ]

    parsed_dates = None
    for fmt in date_formats_to_try:
        try:
            parsed_dates = pd.to_datetime(df['message_date'], format=fmt, errors='raise')
            break
        except (ValueError, TypeError):
            continue

    if parsed_dates is None:
        # Graceful fallback: per-item parsing with dateutil/pandas parser
        parsed_dates = pd.to_datetime(df['message_date'], errors='coerce', dayfirst=True)

    df['date'] = parsed_dates
    df.dropna(subset=['date'], inplace=True)
    df.drop(columns=['message_date'], inplace=True)

    if df.empty:
        raise ValueError("Failed to parse timestamps from the provided chat export.")

    # -------------------------------------------------------------------------
    # Extract Author & Raw Message
    # -------------------------------------------------------------------------
    users = []
    messages = []
    is_system_list = []

    for raw in df['user_message']:
        raw_str = raw.strip()

        # WhatsApp format: "Author: Message"
        # We split on first ": " where the author name does not contain newlines and is reasonably sized
        parts = raw_str.split(': ', 1)
        if len(parts) == 2 and '\n' not in parts[0] and len(parts[0]) <= 60:
            author = parts[0].strip()
            msg_body = parts[1].strip()
            users.append(author)
            messages.append(msg_body)
            is_system_list.append(False)
        else:
            # Check if this is a known system notification
            lower_msg = raw_str.lower()
            is_sys = any(indicator in lower_msg for indicator in SYSTEM_INDICATORS)
            users.append('System')
            messages.append(raw_str)
            is_system_list.append(is_sys)

    df['user'] = users
    df['message'] = messages
    df['is_system'] = is_system_list
    df.drop(columns=['user_message'], inplace=True)

    # Flag media and deleted messages
    df['is_media'] = df['message'].apply(lambda m: bool(MEDIA_PATTERN.search(m)))
    df['is_deleted'] = df['message'].apply(lambda m: bool(DELETED_PATTERN.search(m)))

    # -------------------------------------------------------------------------
    # Feature Engineering (Temporal Dimensions)
    # -------------------------------------------------------------------------
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Formatted hourly period (e.g., "09-10", "23-00")
    df['period'] = df['hour'].apply(lambda h: f"{h:02d}-{(h + 1) % 24:02d}")

    df.reset_index(drop=True, inplace=True)
    return df
