package feasibilityCheck.util;

import java.io.File;

public class DeleteDirectory {
    public static void deleteDirectory(String path) {
        File dir = new File(path);
        if (dir.exists()) {
            File[] subFiles = dir.listFiles();
            for (File subFile : subFiles) {
                if (subFile.isDirectory()) {
                    deleteDirectory(subFile.getPath());
                } else {
                    subFile.delete();
                }
            }
            dir.delete();
        }
    }
}
