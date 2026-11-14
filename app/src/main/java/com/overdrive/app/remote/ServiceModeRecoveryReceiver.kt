package com.overdrive.app.remote

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.SystemClock
import android.provider.Settings
import android.util.Log
import com.overdrive.app.config.UnifiedConfigManager
import org.json.JSONObject

/**
 * App-process safety net for the optional launcher alias hidden by Service Mode.
 * The daemon remains the authority for restoring third-party packages, but this
 * receiver can restore OverDrive's own launcher entry even when that daemon is
 * stopped or fails to start. The deadline is monotonic during a boot; after a
 * reboot it is rebased from the persisted local-date epoch deadline.
 */
class ServiceModeRecoveryReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent?) {
        val pending = goAsync()
        Thread({
            try {
                reconcile(context.applicationContext)
            } catch (error: Throwable) {
                Log.e(TAG, "Service Mode launcher recovery failed", error)
            } finally {
                pending.finish()
            }
        }, "service-mode-launcher-recovery").start()
    }

    companion object {
        private const val TAG = "ServiceModeRecovery"
        private const val ACTION_RECOVER = "com.overdrive.app.action.SERVICE_MODE_RECOVER"
        private const val REQUEST_CODE = 697
        private const val LAUNCHER_CLASS = "com.overdrive.app.LauncherActivity"
        private const val PREFS_NAME = "service_mode_launcher_recovery"
        private const val PREF_BOOT_COUNT = "boot_count"
        private const val PREF_DEADLINE_ELAPSED = "deadline_elapsed_realtime_ms"

        /** Called by the app-process bridge before the launcher alias is disabled. */
        @JvmStatic
        fun schedule(
            context: Context,
            endsAtEpochMs: Long,
            endsAtElapsedRealtimeMs: Long,
            bootCount: Int,
        ) {
            val nowBoot = currentBootCount(context)
            val remainingMs = (endsAtEpochMs - System.currentTimeMillis()).coerceAtLeast(0L)
            val deadline = if (endsAtElapsedRealtimeMs > 0L && nowBoot >= 0 && bootCount == nowBoot) {
                endsAtElapsedRealtimeMs
            } else {
                SystemClock.elapsedRealtime() + remainingMs
            }
            persistDeadline(context, nowBoot, if (nowBoot >= 0) deadline else 0L)
            scheduleAt(context, deadline)
        }

        /** Cancelled when the daemon restores the alias manually. */
        @JvmStatic
        fun cancel(context: Context) {
            val alarm = context.getSystemService(Context.ALARM_SERVICE) as? AlarmManager
            if (alarm != null) alarm.cancel(pendingIntent(context))
            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .clear()
                .commit()
        }

        /** Asynchronous entry point used by MainActivity as a last-resort recovery path. */
        @JvmStatic
        fun recoverNow(context: Context) {
            val appContext = context.applicationContext
            Thread({
                try {
                    reconcile(appContext)
                } catch (error: Throwable) {
                    Log.e(TAG, "Service Mode launcher recovery failed", error)
                }
            }, "service-mode-launcher-recovery-now").start()
        }

        private fun reconcile(context: Context) {
            UnifiedConfigManager.forceReload()
            val state = UnifiedConfigManager.loadConfig().optJSONObject("serviceMode") ?: JSONObject()
            val enabled = state.optBoolean("enabled", false)
            val ending = state.optBoolean("ending", false)
            val hideOverdrive = state.optBoolean("hideOverdrive", false)
            val originalState = state.optInt("launcherOriginalState", -1)

            if (!enabled || ending || !hideOverdrive) {
                if (originalState >= 0) restoreLauncher(context, originalState)
                cancel(context)
                return
            }

            val endsAtEpochMs = state.optLong("endsAtEpochMs", 0L)
            val endsAtElapsedRealtimeMs = state.optLong("endsAtElapsedRealtimeMs", 0L)
            val stateBootCount = state.optInt("bootCount", -1)
            val currentBoot = currentBootCount(context)
            val deadline = resolveDeadline(
                context,
                endsAtEpochMs,
                endsAtElapsedRealtimeMs,
                stateBootCount,
                currentBoot,
            )
            val expired = if (currentBoot >= 0 && deadline > 0L) {
                SystemClock.elapsedRealtime() >= deadline
            } else {
                endsAtEpochMs <= System.currentTimeMillis()
            }

            if (expired) {
                restoreLauncher(context, originalState)
                cancel(context)
                return
            }

            // A package update or OEM component reset may have re-enabled the alias
            // during an active session. Ensure the fallback alarm is durable before
            // disabling the launcher entry again.
            schedule(context, endsAtEpochMs, deadline, currentBoot)
            setLauncherState(context, PackageManager.COMPONENT_ENABLED_STATE_DISABLED)
        }

        private fun restoreLauncher(context: Context, originalState: Int) {
            val safeState = if (originalState in
                PackageManager.COMPONENT_ENABLED_STATE_DEFAULT..PackageManager.COMPONENT_ENABLED_STATE_DISABLED_UNTIL_USED
            ) originalState else PackageManager.COMPONENT_ENABLED_STATE_DEFAULT
            setLauncherState(context, safeState)
        }

        private fun setLauncherState(context: Context, state: Int) {
            val launcher = ComponentName(context, LAUNCHER_CLASS)
            context.packageManager.setComponentEnabledSetting(
                launcher,
                state,
                PackageManager.DONT_KILL_APP,
            )
        }

        private fun currentBootCount(context: Context): Int = try {
            Settings.Global.getInt(context.contentResolver, Settings.Global.BOOT_COUNT, -1)
        } catch (_: Throwable) {
            -1
        }

        /** Keep a rebased monotonic deadline in app storage if the daemon is unavailable. */
        private fun resolveDeadline(
            context: Context,
            endsAtEpochMs: Long,
            stateDeadlineElapsedRealtimeMs: Long,
            stateBootCount: Int,
            currentBoot: Int,
        ): Long {
            if (currentBoot < 0) return 0L
            val preferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val savedBoot = preferences.getInt(PREF_BOOT_COUNT, -1)
            val savedDeadline = preferences.getLong(PREF_DEADLINE_ELAPSED, 0L)
            if (savedBoot == currentBoot && savedDeadline > 0L) return savedDeadline

            val deadline = if (stateBootCount == currentBoot && stateDeadlineElapsedRealtimeMs > 0L) {
                stateDeadlineElapsedRealtimeMs
            } else {
                SystemClock.elapsedRealtime() + (endsAtEpochMs - System.currentTimeMillis()).coerceAtLeast(0L)
            }
            persistDeadline(context, currentBoot, deadline)
            return deadline
        }

        private fun persistDeadline(context: Context, bootCount: Int, deadlineElapsedRealtimeMs: Long) {
            val saved = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putInt(PREF_BOOT_COUNT, bootCount)
                .putLong(PREF_DEADLINE_ELAPSED, deadlineElapsedRealtimeMs)
                .commit()
            if (!saved) throw IllegalStateException("Could not persist Service Mode recovery deadline")
        }

        private fun scheduleAt(context: Context, deadlineElapsedRealtimeMs: Long) {
            val alarm = context.getSystemService(Context.ALARM_SERVICE) as? AlarmManager ?: return
            val triggerAt = deadlineElapsedRealtimeMs.coerceAtLeast(SystemClock.elapsedRealtime() + 1000L)
            val pending = pendingIntent(context)
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    alarm.setExactAndAllowWhileIdle(AlarmManager.ELAPSED_REALTIME_WAKEUP, triggerAt, pending)
                } else {
                    alarm.setExact(AlarmManager.ELAPSED_REALTIME_WAKEUP, triggerAt, pending)
                }
            } catch (error: SecurityException) {
                Log.w(TAG, "Exact recovery alarm unavailable; using an idle-allowed alarm", error)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    alarm.setAndAllowWhileIdle(AlarmManager.ELAPSED_REALTIME_WAKEUP, triggerAt, pending)
                } else {
                    alarm.set(AlarmManager.ELAPSED_REALTIME_WAKEUP, triggerAt, pending)
                }
            }
        }

        private fun pendingIntent(context: Context): PendingIntent {
            val intent = Intent(context, ServiceModeRecoveryReceiver::class.java).setAction(ACTION_RECOVER)
            val flags = PendingIntent.FLAG_UPDATE_CURRENT or
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_IMMUTABLE else 0
            return PendingIntent.getBroadcast(context, REQUEST_CODE, intent, flags)
        }
    }
}
