Файлы:
- yadisk_simple_sync_common.py
- yadisk_simple_upload.py
- yadisk_simple_download.py
- sync.sh

Загрузка:
    python yadisk_simple_upload.py --local "/path/to/local" --remote-root "disk:/backup" --token "TOKEN"

Скачивание:
    python yadisk_simple_download.py --local "/path/to/local" --remote-root "disk:/backup" --token "TOKEN"

Чтобы перезаписывать обычные файлы с тем же именем, но другим содержимым:
    --overwrite-different-md5

Рекомендуется использовать sync.sh для синхронизации ./runs/ :
    ./sync.sh download OPTIONAL_SUBDIR
    ./sync.sh upload OPTIONAL_SUBDIR

Правила для сравнения:
- обычные файлы считаются совпавшими, если совпадает md5;
- если удалённый md5 недоступен, файл считается несовпавшим.

Правила для plots:
- локальная папка с именем plots загружается как plots.zip;
- на Яндекс Диске такие zip-файлы хранятся как *.zipfast;
- если всё кроме plots.zip в соответствующем subtree уже идеально синхронизировано и удалённый plots.zip существует, upload считает plots синхронизированным без локальной сборки zip;
- иначе upload архивирует локальный plots/, сравнивает md5 с удалённым plots.zip и при необходимости загружает его;
- для download схожая логика: если всё кроме plots уже синхронизировано и локальный plots существует, то download считает plots синхронизированным и не скачивает
- иначе скачивает plots.zip и разархивирует

