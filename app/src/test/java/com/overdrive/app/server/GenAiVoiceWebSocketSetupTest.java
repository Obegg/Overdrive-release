package com.overdrive.app.server;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.overdrive.app.genai.GenAiAction;
import com.overdrive.app.genai.GenAiContext;
import com.overdrive.app.genai.GenAiVoiceCommand;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;

/**
 * Validates the Gemini Live {@code BidiGenerateContentSetup} payload the bridge sends against the
 * documented constraints (ai.google.dev/api/live, /api/generate-content#Schema, the gemini-3.8-live
 * migration notes) and against the two tools that reached {@code setupComplete} on the car in
 * v53.1: the far-field tool may not use any schema keyword those two do not use.
 */
public class GenAiVoiceWebSocketSetupTest {

    /** Documented BidiGenerateContentSetup fields. */
    private static final Set<String> SETUP_FIELDS = new HashSet<>(Arrays.asList(
            "model", "generationConfig", "systemInstruction", "tools", "realtimeInputConfig",
            "sessionResumption", "contextWindowCompression", "inputAudioTranscription",
            "outputAudioTranscription", "proactivity"));

    /** Documented FunctionDeclaration fields the Live API accepts. */
    private static final Set<String> DECLARATION_FIELDS = new HashSet<>(Arrays.asList(
            "name", "description", "parameters", "behavior"));

    /**
     * Documented Gemini Schema fields (an OpenAPI 3.0 subset), plus {@code additionalProperties},
     * which the two field-proven tools send and which stays consistent across all three tools.
     */
    // Gemini's Schema proto only. "additionalProperties" is deliberately absent: the 2026-10-10
    // field log shows Gemini Live closing with 1007 'Unknown name "additionalProperties"'.
    private static final Set<String> SCHEMA_FIELDS = new HashSet<>(Arrays.asList(
            "type", "format", "title", "description", "nullable", "enum", "maxItems", "minItems",
            "properties", "required", "minProperties", "maxProperties", "minLength", "maxLength",
            "pattern", "example", "anyOf", "propertyOrdering", "default", "items", "minimum",
            "maximum"));

    private static final Set<String> SCHEMA_TYPES = new HashSet<>(Arrays.asList(
            "object", "string", "integer", "number", "boolean", "array"));

    @Test
    public void setupUsesOnlyDocumentedFieldsAndNoneThe38LiveModelRejects() throws Exception {
        for (boolean farField : new boolean[] {false, true}) {
            JSONObject setup = GenAiVoiceWebSocket.geminiSetup("gemini-3.8-live", "grounded", farField)
                    .getJSONObject("setup");
            for (String key : keys(setup)) assertTrue(key, SETUP_FIELDS.contains(key));
            assertEquals("models/gemini-3.8-live", setup.getString("model"));
            JSONObject generation = setup.getJSONObject("generationConfig");
            assertEquals("AUDIO", generation.getJSONArray("responseModalities").getString(0));
            assertEquals(1, generation.getJSONArray("responseModalities").length());
            assertFalse(generation.has("thinkingConfig"));
            assertFalse(generation.has("thinking_level"));
            // Removed from the API (affective dialogue) or permanently on (proactive audio) on 3.8.
            assertFalse(setup.has("proactivity"));
            String text = setup.toString();
            assertFalse(text.contains("enableAffectiveDialog"));
            assertFalse(text.contains("enable_affective_dialog"));
            assertFalse(text.contains("proactiveAudio"));
            assertFalse(text.contains("thinking"));
            assertTrue(setup.has("outputAudioTranscription"));
            // Hands-free only: tuned automatic VAD, input transcription and goAway resumption.
            assertEquals(farField, setup.has("inputAudioTranscription"));
            assertEquals(farField, setup.has("sessionResumption"));
            assertEquals(farField, setup.has("realtimeInputConfig"));
            if (farField) {
                JSONObject vad = setup.getJSONObject("realtimeInputConfig")
                        .getJSONObject("automaticActivityDetection");
                assertEquals(new HashSet<>(Arrays.asList("startOfSpeechSensitivity", "endOfSpeechSensitivity",
                        "prefixPaddingMs", "silenceDurationMs")), new HashSet<>(keys(vad)));
                assertEquals("START_SENSITIVITY_LOW", vad.getString("startOfSpeechSensitivity"));
                // The engine's audioStreamEnd decides; the server's slow end is only the fallback.
                assertEquals("END_SENSITIVITY_LOW", vad.getString("endOfSpeechSensitivity"));
                assertTrue(vad.getInt("silenceDurationMs") > 2500);
                assertEquals(0, setup.getJSONObject("sessionResumption").length());
            }
            assertEquals(1, setup.getJSONArray("tools").length());
        }
        // The resource prefix is not doubled.
        assertEquals("models/gemini-3.1-flash-live-preview",
                GenAiVoiceWebSocket.geminiSetup("models/gemini-3.1-flash-live-preview", "x")
                        .getJSONObject("setup").getString("model"));
    }

    @Test
    public void everyDeclarationIsBlockingAndWellFormed() throws Exception {
        JSONArray declarations = declarations(true);
        assertEquals(3, declarations.length());
        assertEquals(2, declarations(false).length());
        for (int i = 0; i < declarations.length(); i++) {
            JSONObject declaration = declarations.getJSONObject(i);
            for (String key : keys(declaration)) assertTrue(key, DECLARATION_FIELDS.contains(key));
            assertTrue(declaration.getString("name").matches("[a-zA-Z_][a-zA-Z0-9_-]{0,63}"));
            assertFalse(declaration.getString("description").trim().isEmpty());
            // gemini-3.8-live defaults to NON_BLOCKING; the bridge needs the model to wait.
            assertEquals(GenAiVoiceWebSocket.GEMINI_TOOL_BEHAVIOR, declaration.getString("behavior"));
            assertEquals("BLOCKING", declaration.getString("behavior"));
            validateSchema(declaration.getString("name"), declaration.getJSONObject("parameters"), true);
        }
    }

    @Test
    public void behaviorIsOnlySentToTheLiveFamilyThatDefaultsToNonBlocking() throws Exception {
        assertTrue(GenAiVoiceWebSocket.geminiDeclaresBehavior("gemini-3.8-live"));
        assertTrue(GenAiVoiceWebSocket.geminiDeclaresBehavior("models/gemini-3.8-live-extended-thinking"));
        assertFalse(GenAiVoiceWebSocket.geminiDeclaresBehavior("gemini-3.1-flash-live-preview"));
        assertFalse(GenAiVoiceWebSocket.geminiDeclaresBehavior(null));
        // The 3.1 preview keeps the exact declaration shape that reached setupComplete in the field.
        JSONArray preview = GenAiVoiceWebSocket.geminiSetup("gemini-3.1-flash-live-preview", "grounded", true)
                .getJSONObject("setup").getJSONArray("tools").getJSONObject(0)
                .getJSONArray("functionDeclarations");
        assertEquals(3, preview.length());
        for (int i = 0; i < preview.length(); i++) {
            assertFalse(preview.getJSONObject(i).has("behavior"));
        }
    }

    @Test
    public void farFieldToolUsesOnlyTheKeywordsTheFieldProvenToolsUse() throws Exception {
        Set<String> proven = new TreeSet<>();
        proven.addAll(keywords(GenAiAction.geminiTool().getJSONObject("parameters")));
        proven.addAll(keywords(GenAiContext.geminiRealtimeTool().getJSONObject("parameters")));
        Set<String> voice = keywords(GenAiVoiceCommand.geminiTool().getJSONObject("parameters"));
        assertTrue("new keywords: " + diff(voice, proven), proven.containsAll(voice));
        // The keyword set the three raw tool objects share; the setup builder then strips what
        // Gemini does not know (see the next test), so this is the pre-strip inventory.
        assertEquals(new TreeSet<>(Arrays.asList(
                "additionalProperties", "enum", "maxLength", "maximum", "minimum", "properties",
                "required", "type")), proven);
    }

    @Test
    public void setupNeverCarriesAdditionalPropertiesWhichGeminiRejectsWith1007() throws Exception {
        for (String model : new String[] {"gemini-3.1-flash-live-preview", "gemini-3.8-live"}) {
            for (boolean farField : new boolean[] {false, true}) {
                String payload = GenAiVoiceWebSocket.geminiSetup(model, "grounded", farField).toString();
                assertFalse(model + " farField=" + farField, payload.contains("additionalProperties"));
                JSONArray declarations = GenAiVoiceWebSocket.geminiSetup(model, "grounded", farField)
                        .getJSONObject("setup").getJSONArray("tools").getJSONObject(0)
                        .getJSONArray("functionDeclarations");
                for (int i = 0; i < declarations.length(); i++) {
                    JSONObject parameters = declarations.getJSONObject(i).getJSONObject("parameters");
                    validateSchema(declarations.getJSONObject(i).getString("name"), parameters, true);
                    // Everything else survives the strip: required list, properties, enums, bounds.
                    assertTrue(parameters.has("required"));
                    assertTrue(parameters.getJSONObject("properties").length() > 0);
                }
            }
        }
        // The strip is recursive and tolerant of absent branches.
        JSONObject nested = new JSONObject()
                .put("type", "object").put("additionalProperties", false)
                .put("properties", new JSONObject()
                        .put("inner", new JSONObject().put("type", "object").put("additionalProperties", false)
                                .put("properties", new JSONObject()
                                        .put("leaf", new JSONObject().put("type", "string"))))
                        .put("list", new JSONObject().put("type", "array")
                                .put("items", new JSONObject().put("type", "object").put("additionalProperties", false))));
        GenAiVoiceWebSocket.stripUnsupportedGeminiSchemaKeys(nested);
        assertFalse(nested.toString().contains("additionalProperties"));
        assertEquals("string", nested.getJSONObject("properties").getJSONObject("inner")
                .getJSONObject("properties").getJSONObject("leaf").getString("type"));
        GenAiVoiceWebSocket.stripUnsupportedGeminiSchemaKeys(null);
        // The OpenAI strict schemas are untouched by the Gemini strip.
        assertFalse(GenAiAction.openAiTool().getJSONObject("parameters").getBoolean("additionalProperties"));
        assertFalse(GenAiVoiceCommand.openAiTool().getJSONObject("parameters").getBoolean("additionalProperties"));
    }

    @Test
    public void geminiVariantKeepsTheOpenAiSemanticsWithoutPropertyDescriptions() throws Exception {
        JSONObject openAi = GenAiVoiceCommand.openAiTool().getJSONObject("parameters");
        JSONObject gemini = GenAiVoiceCommand.geminiTool().getJSONObject("parameters");
        assertEquals(openAi.getJSONArray("required").toString(), gemini.getJSONArray("required").toString());
        JSONObject openAiProperties = openAi.getJSONObject("properties");
        JSONObject geminiProperties = gemini.getJSONObject("properties");
        assertEquals(keys(openAiProperties), keys(geminiProperties));
        String guide = GenAiVoiceCommand.geminiTool().getString("description");
        for (String name : keys(openAiProperties)) {
            JSONObject a = openAiProperties.getJSONObject(name);
            JSONObject g = geminiProperties.getJSONObject(name);
            String description = a.getString("description");
            assertFalse(name, g.has("description"));
            // Same type, enum and bounds once the description is set aside.
            a.remove("description");
            assertEquals(name, keys(a), keys(g));
            for (String key : keys(a)) assertEquals(name + "." + key, a.get(key).toString(), g.get(key).toString());
            // The guidance moved into the tool description, verbatim.
            assertTrue(name, guide.contains(name + ": " + description));
        }
        // The OpenAI strict schema is untouched (fresh instance: the loop above edited its copy).
        JSONObject strict = GenAiVoiceCommand.openAiTool();
        JSONObject strictParameters = strict.getJSONObject("parameters");
        assertFalse(strictParameters.getBoolean("additionalProperties"));
        for (String name : keys(strictParameters.getJSONObject("properties"))) {
            assertTrue(name, strictParameters.getJSONObject("properties").getJSONObject(name).has("description"));
        }
        assertFalse(strict.getString("description").contains("Fields:"));
    }

    @Test
    public void payloadStaysSmallAndFreeOfTranscriptionWording() throws Exception {
        String text = GenAiVoiceWebSocket.geminiSetup("gemini-3.8-live", "grounded", true).toString();
        assertFalse(text.contains("transcription\""));   // only the *AudioTranscription keys
        assertTrue(text.length() < 16_000);
    }

    // ---- helpers ----

    private static JSONArray declarations(boolean farField) throws Exception {
        return GenAiVoiceWebSocket.geminiSetup("gemini-3.8-live", "grounded", farField)
                .getJSONObject("setup").getJSONArray("tools").getJSONObject(0)
                .getJSONArray("functionDeclarations");
    }

    /** Recursive Schema validation against the documented field list and type rules. */
    private static void validateSchema(String path, JSONObject schema, boolean root) throws Exception {
        for (String key : keys(schema)) assertTrue(path + "." + key, SCHEMA_FIELDS.contains(key));
        String type = schema.getString("type");
        assertTrue(path + " type " + type, SCHEMA_TYPES.contains(type));
        if (root) assertEquals(path, "object", type);
        if (schema.has("enum")) {
            assertEquals(path + " enum needs a string type", "string", type);
            JSONArray values = schema.getJSONArray("enum");
            assertTrue(values.length() > 0);
            for (int i = 0; i < values.length(); i++) assertTrue(values.get(i) instanceof String);
        }
        if (schema.has("minimum") || schema.has("maximum")) {
            assertTrue(path + " bounds need a numeric type", "integer".equals(type) || "number".equals(type));
            if (schema.has("minimum") && schema.has("maximum")) {
                assertTrue(path, schema.getDouble("minimum") <= schema.getDouble("maximum"));
            }
        }
        if (schema.has("maxLength")) assertEquals(path, "string", type);
        if (schema.has("properties")) {
            assertEquals(path, "object", type);
            JSONObject properties = schema.getJSONObject("properties");
            Set<String> names = keys(properties);
            assertTrue(path + " has no properties", !names.isEmpty());
            for (String name : names) {
                assertTrue(name, name.matches("[a-zA-Z_][a-zA-Z0-9_]{0,63}"));
                validateSchema(path + "." + name, properties.getJSONObject(name), false);
            }
            if (schema.has("required")) {
                JSONArray required = schema.getJSONArray("required");
                for (int i = 0; i < required.length(); i++) {
                    assertTrue(path + " requires unknown " + required.getString(i),
                            names.contains(required.getString(i)));
                }
            }
        } else {
            assertFalse(path + " required without properties", schema.has("required"));
        }
    }

    private static Set<String> keywords(JSONObject schema) throws Exception {
        Set<String> out = new TreeSet<>();
        collectKeywords(schema, out);
        return out;
    }

    private static void collectKeywords(JSONObject schema, Set<String> out) throws Exception {
        for (String key : keys(schema)) {
            out.add(key);
            if ("properties".equals(key)) {
                JSONObject properties = schema.getJSONObject(key);
                for (String name : keys(properties)) collectKeywords(properties.getJSONObject(name), out);
            } else if ("items".equals(key) && schema.opt(key) instanceof JSONObject) {
                collectKeywords(schema.getJSONObject(key), out);
            }
        }
    }

    private static Set<String> keys(JSONObject object) {
        Set<String> out = new TreeSet<>();
        for (Iterator<String> it = object.keys(); it.hasNext(); ) out.add(it.next());
        return out;
    }

    private static Set<String> diff(Set<String> a, Set<String> b) {
        Set<String> out = new TreeSet<>(a);
        out.removeAll(b);
        return out;
    }
}
