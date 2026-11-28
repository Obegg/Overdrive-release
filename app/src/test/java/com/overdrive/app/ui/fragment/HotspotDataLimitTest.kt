package com.overdrive.app.ui.fragment

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class HotspotDataLimitTest {
    @Test fun blankAndWholeNumbersAreAccepted() {
        assertEquals(0L, parseHotspotDataLimit(""))
        assertEquals(0L, parseHotspotDataLimit("0"))
        assertEquals(128L, parseHotspotDataLimit("00128"))
        assertEquals(9_007_199_254_740_991L, parseHotspotDataLimit("9007199254740991"))
    }

    @Test fun invalidAndUnrepresentableValuesAreRejected() {
        for (value in listOf("-1", "+1", "1.5", "1e3", "abc", "9007199254740992", "999999999999999999999")) {
            assertNull(value, parseHotspotDataLimit(value))
        }
    }
}
