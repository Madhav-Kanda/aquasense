package aqua.analyses;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;

public class PyCompilerRunner {
    public static void main (String[] args){

        if (args[0].endsWith("template")) {
            PytorchCompiler pytorchCompiler = new PytorchCompiler();
            String templatePath = args[0];
            String pytorchCode = pytorchCompiler.runCompiler(templatePath);
            try {
                // tempfilePath = File.createTempFile(tempFileName, ".template");
                File torchFile = new File(templatePath.substring(0, templatePath.length() - 9) + ".py");
                FileUtils.writeStringToFile(torchFile, pytorchCode);
                System.out.println("Pytorch Code in " + (templatePath.substring(0, templatePath.length() - 9) + ".py"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        else {
            PytorchCompiler pytorchCompiler = new PytorchCompiler();
            String stanDirPath = args[0];
            if (stanDirPath.endsWith("/")) {
                stanDirPath = stanDirPath.substring(0, stanDirPath.length() - 1);
            }
            int index0=stanDirPath.lastIndexOf('/');
            String stanName = stanDirPath.substring(index0+1,stanDirPath.length());
            String pytorchCode = pytorchCompiler.fromStan(stanDirPath);
            try {
                // tempfilePath = File.createTempFile(tempFileName, ".template");
                File torchFile = new File(stanDirPath + "/" + stanName +  ".py");
                System.out.println("Pytorch Code in " + stanDirPath + "/" + stanName +  ".py");
                FileUtils.writeStringToFile(torchFile, pytorchCode);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
