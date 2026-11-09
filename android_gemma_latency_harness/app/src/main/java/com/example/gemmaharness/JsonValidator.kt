package com.example.gemmaharness

import org.json.JSONObject
import org.json.JSONException

object JsonValidator {

    data class ValidationResult(
        val isClosed: Boolean,
        val parseSuccess: Boolean,
        val fieldsPresent: Boolean
    )

    fun validate(jsonString: String): ValidationResult {
        val trimmed = jsonString.trim()
        val isClosed = trimmed.startsWith("{") && trimmed.endsWith("}")

        if (!isClosed) {
            return ValidationResult(isClosed = false, parseSuccess = false, fieldsPresent = false)
        }

        var parseSuccess = false
        var fieldsPresent = false
        try {
            val jsonObject = JSONObject(trimmed)
            parseSuccess = true // If it doesn't throw, parsing was successful

            val hasCommentType = jsonObject.has("comment_type")
            val hasShortReason = jsonObject.has("short_reason")
            val hasDetailedReason = jsonObject.has("detailed_reason")
            fieldsPresent = hasCommentType && hasShortReason && hasDetailedReason

        } catch (e: JSONException) {
            parseSuccess = false
            fieldsPresent = false
        }

        return ValidationResult(isClosed, parseSuccess, fieldsPresent)
    }
}
