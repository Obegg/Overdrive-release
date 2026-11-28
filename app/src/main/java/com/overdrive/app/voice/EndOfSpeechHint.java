package com.overdrive.app.voice;

import java.util.Locale;

/**
 * Whether the provider's transcript of the user's turn so far ends "unfinished" (a trailing
 * conjunction, preposition, article, auxiliary, filler or command verb, a comma, or at most one word
 * after the wake word), so the end of speech waits a little longer. Pure string checks; English and
 * Chinese only, anything else never extends.
 */
final class EndOfSpeechHint {

    static final int LANG_NONE = 0;
    static final int LANG_EN = 1;
    static final int LANG_ZH = 2;

    /** Class of the transcript's ending; everything but these three means unfinished. */
    static final String COMPLETE = "complete";
    static final String NONE = "none";
    static final String UNSUPPORTED = "unsupported";

    private static final String[][] EN = {
            {"conj", "and", "or", "but", "so", "then"},
            // No "on" or "like": complete commands end on them ("max cooling on", "what's the weather like").
            {"prep", "to", "of", "for", "with", "at", "in", "from", "by"},
            {"article", "the", "a", "an", "my", "your"},
            {"verb", "set", "turn", "change", "make", "put", "open", "close", "play", "call", "navigate"},
            {"aux", "is", "are", "can", "could", "would", "please"},
            {"filler", "um", "uh", "er"},
    };
    private static final String[][] ZH = {
            {"conj", "和", "跟", "还有", "然后"},
            {"prep", "把", "给", "到", "在", "让"},
            {"particle", "的"},
            {"verb", "帮我"},
            {"filler", "嗯", "呃", "那个", "就是"},
    };

    private EndOfSpeechHint() {
    }

    /** The configured command language, else the app locale; LANG_NONE for anything else. */
    static int languageFor(String commandLanguage, String localeTag) {
        String tag = commandLanguage == null || commandLanguage.isEmpty() ? localeTag : commandLanguage;
        if (tag == null) return LANG_NONE;
        String t = tag.toLowerCase(Locale.ROOT);
        if (t.startsWith("en")) return LANG_EN;
        if (t.startsWith("zh")) return LANG_ZH;
        return LANG_NONE;
    }

    /** The ending's class: {@link #COMPLETE}, {@link #NONE} (no transcript), {@link #UNSUPPORTED} or an unfinished class. */
    static String classify(String transcript, int lang) {
        if (lang == LANG_NONE) return UNSUPPORTED;
        if (transcript == null) return NONE;
        String s = transcript.trim();
        if (s.isEmpty()) return NONE;
        char last = s.charAt(s.length() - 1);
        if (last == ',' || last == '，' || last == '、') return "comma";
        return lang == LANG_EN ? english(s) : chinese(s);
    }

    static boolean unfinished(String cls) {
        return !COMPLETE.equals(cls) && !NONE.equals(cls) && !UNSUPPORTED.equals(cls);
    }

    private static String english(String s) {
        // Words without punctuation, lowercase, scanned from the end: no split or regex allocation.
        int end = s.length();
        int words = 0;
        String lastWord = null;
        boolean afterWake = false;
        while (end > 0) {
            while (end > 0 && !Character.isLetterOrDigit(s.charAt(end - 1)) && s.charAt(end - 1) != '\'') end--;
            if (end == 0) break;
            int start = end;
            while (start > 0 && (Character.isLetterOrDigit(s.charAt(start - 1)) || s.charAt(start - 1) == '\'')) start--;
            if (lastWord == null) lastWord = s.substring(start, end).toLowerCase(Locale.ROOT);
            if (isWake(s, start, end)) {
                afterWake = true;
                break;
            }
            words++;
            if (words > 1 && lastWord != null) break;
            end = start;
        }
        if (afterWake && words <= 1) return "short";
        if (lastWord == null) return NONE;
        for (String[] group : EN) {
            for (int i = 1; i < group.length; i++) {
                if (group[i].equals(lastWord)) return group[0];
            }
        }
        return COMPLETE;
    }

    /** "OverDrive" (any case) or "over drive" ending at {@code end}. */
    private static boolean isWake(String s, int start, int end) {
        if (s.regionMatches(true, start, "overdrive", 0, 9) && end - start == 9) return true;
        if (end - start == 5 && s.regionMatches(true, start, "drive", 0, 5)) {
            int p = start;
            while (p > 0 && s.charAt(p - 1) == ' ') p--;
            return p >= 4 && s.regionMatches(true, p - 4, "over", 0, 4);
        }
        return false;
    }

    private static String chinese(String s) {
        int end = s.length();
        while (end > 0 && isPunct(s.charAt(end - 1))) end--;
        if (end == 0) return NONE;
        for (String[] group : ZH) {
            for (int i = 1; i < group.length; i++) {
                String w = group[i];
                if (end >= w.length() && s.startsWith(w, end - w.length())) return group[0];
            }
        }
        return COMPLETE;
    }

    private static boolean isPunct(char c) {
        return c == '。' || c == '！' || c == '？' || c == '.' || c == '!' || c == '?' || c == ' '
                || c == '…' || c == '"' || c == '”';
    }
}
