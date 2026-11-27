package com.overdrive.app.server;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

public class GenAiVoiceWebSocketTest {

    @Test
    public void providerSetupsUseNativeAudioWithoutInputTranscription() throws Exception {
        JSONObject openAi = GenAiVoiceWebSocket.openAiSetup(
                "gpt-realtime-2.1", "grounded");
        JSONObject session = openAi.getJSONObject("session");
        assertEquals("audio",
                session.getJSONArray("output_modalities").getString(0));
        assertEquals(24_000, session.getJSONObject("audio")
                .getJSONObject("input")
                .getJSONObject("format")
                .getInt("rate"));
        assertEquals("semantic_vad", session.getJSONObject("audio")
                .getJSONObject("input")
                .getJSONObject("turn_detection")
                .getString("type"));
        assertEquals("propose_overdrive_action",
                session.getJSONArray("tools")
                        .getJSONObject(0).getString("name"));
        assertEquals("get_overdrive_context",
                session.getJSONArray("tools")
                        .getJSONObject(1).getString("name"));
        assertFalse(openAi.toString().contains("transcription"));

        JSONObject gemini = GenAiVoiceWebSocket.geminiSetup(
                "gemini-3.1-flash-live-preview", "grounded");
        JSONObject setup = gemini.getJSONObject("setup");
        assertEquals("models/gemini-3.1-flash-live-preview",
                setup.getString("model"));
        assertTrue(setup.has("outputAudioTranscription"));
        // Near-field (the Assistant page) keeps the server's default VAD and no input transcript.
        assertFalse(setup.has("inputAudioTranscription"));
        assertFalse(setup.has("realtimeInputConfig"));
        assertEquals("{\"realtimeInput\":{\"audioStreamEnd\":true}}",
                GenAiVoiceWebSocket.geminiAudioStreamEnd().toString());
        assertEquals("propose_overdrive_action",
                setup.getJSONArray("tools").getJSONObject(0)
                        .getJSONArray("functionDeclarations")
                        .getJSONObject(0).getString("name"));
        assertEquals("get_overdrive_context",
                setup.getJSONArray("tools").getJSONObject(0)
                        .getJSONArray("functionDeclarations")
                        .getJSONObject(1).getString("name"));
    }

    @Test
    public void farFieldSetupsAddTheTypedCommandToolAsThirdTool() throws Exception {
        JSONObject openAi = GenAiVoiceWebSocket.openAiSetup(
                "gpt-realtime-2.1", "grounded", true);
        JSONArray openAiTools = openAi.getJSONObject("session")
                .getJSONArray("tools");
        // The fourth is search_web, which every OpenAI session gets.
        assertEquals(4, openAiTools.length());
        assertEquals("propose_overdrive_action",
                openAiTools.getJSONObject(0).getString("name"));
        assertEquals("get_overdrive_context",
                openAiTools.getJSONObject(1).getString("name"));
        assertEquals("run_overdrive_command",
                openAiTools.getJSONObject(2).getString("name"));
        assertFalse(openAi.toString().contains("transcription"));
        JSONObject gemini = GenAiVoiceWebSocket.geminiSetup(
                "gemini-3.1-flash-live-preview", "grounded", true);
        JSONArray declarations = gemini.getJSONObject("setup")
                .getJSONArray("tools").getJSONObject(0)
                .getJSONArray("functionDeclarations");
        assertEquals(3, declarations.length());
        assertEquals("propose_overdrive_action",
                declarations.getJSONObject(0).getString("name"));
        assertEquals("get_overdrive_context",
                declarations.getJSONObject(1).getString("name"));
        assertEquals("run_overdrive_command",
                declarations.getJSONObject(2).getString("name"));
        assertFalse(declarations.toString().contains("transcription"));
    }

    @Test
    public void nearFieldSetupsKeepExactlyTheTwoPageTools() throws Exception {
        for (JSONObject openAi : new JSONObject[] {
                GenAiVoiceWebSocket.openAiSetup("gpt-realtime-2.1", "grounded"),
                GenAiVoiceWebSocket.openAiSetup("gpt-realtime-2.1", "grounded", false)}) {
            JSONArray tools = openAi.getJSONObject("session").getJSONArray("tools");
            // The read-only search_web is the only addition; never the typed command tool.
            assertEquals(3, tools.length());
            assertEquals("propose_overdrive_action",
                    tools.getJSONObject(0).getString("name"));
            assertEquals("get_overdrive_context",
                    tools.getJSONObject(1).getString("name"));
            assertEquals("search_web", tools.getJSONObject(2).getString("name"));
        }
        for (JSONObject gemini : new JSONObject[] {
                GenAiVoiceWebSocket.geminiSetup("gemini-3.1-flash-live-preview", "grounded"),
                GenAiVoiceWebSocket.geminiSetup("gemini-3.1-flash-live-preview", "grounded", false)}) {
            JSONArray declarations = gemini.getJSONObject("setup")
                    .getJSONArray("tools").getJSONObject(0)
                    .getJSONArray("functionDeclarations");
            assertEquals(2, declarations.length());
            assertEquals("propose_overdrive_action",
                    declarations.getJSONObject(0).getString("name"));
            assertEquals("get_overdrive_context",
                    declarations.getJSONObject(1).getString("name"));
        }
    }

    @Test
    public void approvedVoiceContextIsStrictlyBounded() throws Exception {
        StringBuilder large = new StringBuilder();
        for (int i = 0; i < 60_000; i++) large.append('x');

        String bounded = GenAiVoiceWebSocket.boundedToolOutput(
                new JSONObject().put("data", large.toString()));

        assertTrue(bounded.length() <= 48_000);
        assertTrue(new JSONObject(bounded).optBoolean("truncated"));
    }
}
