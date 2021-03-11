package com.quadtalent.quadx.io;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import lombok.SneakyThrows;

import java.io.File;
import java.io.IOException;

/**
 * @author zhnlk
 * @date 2020/7/29
 * @mail yanan.zyn@quadtalent.com
 */
public class JacksonUtils {

    private static ObjectMapper OBJECT_MAPPER;

    static {
        if (OBJECT_MAPPER == null) {
            OBJECT_MAPPER = new ObjectMapper();
            JavaTimeModule timeModule = new JavaTimeModule();
//            timeModule.addDeserializer(LocalDateTime.class, new LocalDateTimeDeserializer());
            OBJECT_MAPPER.registerModule(timeModule);
            OBJECT_MAPPER.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
            OBJECT_MAPPER.enable(DeserializationFeature.ACCEPT_SINGLE_VALUE_AS_ARRAY);
//            OBJECT_MAPPER.enable(SerializationFeature.WRAP_ROOT_VALUE);

            OBJECT_MAPPER.setSerializationInclusion(JsonInclude.Include.NON_EMPTY);
        }
    }

    @SneakyThrows({IllegalArgumentException.class, IOException.class})
    public static <T> T fromString(String string, Class<T> clazz) {
        return OBJECT_MAPPER.readValue(string, clazz);
    }

    @SneakyThrows({IllegalArgumentException.class, IOException.class})
    public static <T> T fromString(String string, TypeReference<T> tr) {
        return OBJECT_MAPPER.readValue(string, tr);
    }

    @SneakyThrows
    public static <T> T fromFile(File file, Class<T> clazz) {
        return OBJECT_MAPPER.readValue(file, clazz);
    }

    @SneakyThrows
    public static <T> T fromFile(File file, TypeReference<T> tr) {
        return OBJECT_MAPPER.readValue(file, tr);
    }

    @SneakyThrows(IOException.class)
    public static void toFile(File file, Object value) {
        OBJECT_MAPPER.writeValue(file, value);
    }

    @SneakyThrows(JsonProcessingException.class)
    public static String toString(Object value) {
        return OBJECT_MAPPER.writeValueAsString(value);
    }

    @SneakyThrows(IOException.class)
    public static JsonNode toJsonNode(String value) {
        return OBJECT_MAPPER.readTree(value);
    }

    public static <T> T clone(T value) {
        return fromString(toString(value), (Class<T>) value.getClass());
    }
}
