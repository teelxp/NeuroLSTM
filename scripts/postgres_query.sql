-- Пример READ-ONLY-запроса. Названия полей согласуйте с существующей схемой.
-- Не добавляйте искусственные mass_out, объем участка или коэффициент Z.
SELECT
    "timestamp" AS "timestamp",
    pressure,
    temperature,
    flow_rate,
    external_temp,
    humidity,
    mass_in,
    mass_out,
    accident
FROM training_data
ORDER BY "timestamp";
-- volume_m3, molar_mass_kg_mol, compressibility можно задать как проверенные
-- константы physics в JSON, если их нет в телеметрии.
