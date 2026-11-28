package com.overdrive.app.ui.fragment

/** A blank value explicitly removes the limit; malformed input must not. */
internal fun parseHotspotDataLimit(raw: String): Long? {
    if (raw.isEmpty()) return 0L
    if (raw.any { it !in '0'..'9' }) return null
    return raw.toLongOrNull()?.takeIf { it <= 9_007_199_254_740_991L }
}
