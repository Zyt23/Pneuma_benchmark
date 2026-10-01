import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.file.metadata.*;
import org.apache.tsfile.read.TsFileSequenceReader;
import org.apache.tsfile.read.common.Chunk;
import org.apache.tsfile.write.writer.TsFileIOWriter;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.*;
import java.util.zip.*;

/** Rebuild device metadata while copying the original compressed numeric chunks.
 *  All original device paths/metadata indices/bloom filters are replaced by
 *  freshly generated metadata. This is not a binary string replacement.
 *  Input/output chunk bytes, schemas, null layouts and relative timestamps are
 *  checked for equality for every file. Private plans must never be published.
 */
public class TsFileDeidentify {
  static final Set<String> STANDARD = names("n21 n22 bmps1 bmps2 precool_press1 precool_press2 prv_eng1_r prv_eng2_r hpv_eng1_r hpv_eng2_r precool_temp1 precool_temp2 pack1_ram_i_dr pack1_ram_o_dr pack2_ram_i_dr pack2_ram_o_dr flight_phase");
  static final Set<String> B787 = names("alt_std ias flight_phase caslcac1oupresmp_01 caslcac1outempmp_01 caslcac1rpmpm_ctl_01 caslcac1surgemp_c_01 caslcac2oupresmp_01 caslcac2outempmp_01 caslcac2rpmpm_ctl_01 caslcac2surgemp_c_01 caslpackcotempmp_01 casrcac1oupresmp_01 casrcac1outempmp_01 casrcac1rpmpm_ctl_01 casrcac1surgemp_c_01 casrcac2oupresmp_01 casrcac2outempmp_01 casrcac2rpmpm_ctl_01 casrcac2surgemp_c_01 casrpackcotempmp_01");
  static final Set<String> B777 = names("n21 n22 flight_phase e60031500l e60031500r e60041500l e60041500r e60091515l e60091515r");
  static Set<String> names(String s) { return new HashSet<>(Arrays.asList(s.split(" "))); }
  static class Plan {
    String source, target, device;
    Plan(String line) {
      String[] v = line.split("\t", -1);
      if (v.length != 3) throw new IllegalArgumentException("Expected three plan columns");
      source=v[0]; target=v[1]; device=v[2];
      if (!target.matches("tsfile_datasets/dataset[0-9]+(-[0-9]+)?_tsfile/[01]/sample_20[0-9]{2}-[0-9_-]+\\.tsfile") || !device.matches("sample_[0-9]{6}"))
        throw new IllegalArgumentException("Invalid pseudonym target");
    }
  }
  static class Info {
    List<ChunkMetadata> chunks = new ArrayList<>();
    String device;
    long points, first=Long.MAX_VALUE, last=Long.MIN_VALUE;
    int channels;
  }
  static Info inspect(TsFileSequenceReader r, String target) throws Exception {
    Map<IDeviceID,List<TimeseriesMetadata>> all = r.getAllTimeseriesMetadata(true);
    if (all.size()!=1) throw new IllegalArgumentException("Only single-device flights are supported");
    Info info=new Info();
    Set<String> allowed=target.contains("/dataset13_") ? B787 : target.contains("/dataset14_") ? B777 : STANDARD;
    for (Map.Entry<IDeviceID,List<TimeseriesMetadata>> e : all.entrySet()) {
      info.device=e.getKey().toString();
      for (TimeseriesMetadata m : e.getValue()) {
        String name=m.getMeasurementId();
        for (ChunkMetadata c : r.readChunkMetaDataList(m)) {
          if (name.isEmpty()) {
            if (c.getDataType()!=TSDataType.VECTOR) throw new IllegalArgumentException("Unexpected time-column type");
            if (c.getNumOfPoints()>1) {
              double step=(c.getEndTime()-c.getStartTime())/(double)(c.getNumOfPoints()-1);
              if (step<100_000_000d || step>10_000_000_000d)
                throw new IllegalArgumentException("Unapproved timestamp scale");
            }
          } else if (!allowed.contains(name) || !Arrays.asList(TSDataType.FLOAT, TSDataType.DOUBLE, TSDataType.INT32, TSDataType.INT64, TSDataType.BOOLEAN).contains(c.getDataType())) {
            throw new IllegalArgumentException("Unapproved measurement or nonnumeric field: "+name);
          }
          info.chunks.add(c);
          if (c.getNumOfPoints()>0) {
            info.first=Math.min(info.first,c.getStartTime());
            info.last=Math.max(info.last,c.getEndTime());
            info.points+=c.getNumOfPoints();
          }
        }
        if (!name.isEmpty()) info.channels++;
      }
    }
    // These QAR files use relative nanosecond ticks. Some retained slices
    // begin later within the source flight; retain that offset unchanged.
    // Fail visibly if a different timestamp representation is encountered.
    if (info.first<0 || info.first>info.last || info.last>31_536_000_000_000_000L
        || info.last-info.first>259_200_000_000_000L)
      throw new IllegalArgumentException("Timestamp range is not the approved relative-nanosecond profile: "+target+" first="+info.first+" last="+info.last);
    info.chunks.sort(Comparator.comparingLong(ChunkMetadata::getOffsetOfChunkHeader));
    return info;
  }
  static void digest(MessageDigest d, ChunkMetadata m, Chunk c) {
    String meta=m.getMeasurementUid()+"\t"+m.getDataType()+"\t"+m.getMask()+"\t"+m.getNumOfPoints()+"\t"+m.getStartTime()+"\t"+m.getEndTime()+"\n";
    d.update(meta.getBytes(StandardCharsets.UTF_8));
    ByteBuffer bytes=c.getData().duplicate();
    d.update(ByteBuffer.allocate(4).putInt(bytes.remaining()).array());
    d.update(bytes);
  }
  static String hex(byte[] data) {
    StringBuilder out=new StringBuilder();
    for (byte b:data) out.append(String.format("%02x", b & 255));
    return out.toString();
  }
  static class Converted {
    Plan plan; byte[] bytes; String audit;
    Converted(Plan p, byte[] b, String a) {plan=p; bytes=b; audit=a;}
  }
  static Converted convert(ZipFile zip, Plan p, java.nio.file.Path work) throws Exception {
    java.nio.file.Path source=Files.createTempFile(work,"input_", ".tsfile");
    java.nio.file.Path output=Files.createTempFile(work,"output_", ".tsfile");
    Files.delete(output);
    try {
      ZipEntry entry=zip.getEntry(p.source);
      if (entry==null) throw new FileNotFoundException("Plan input is absent");
      try (InputStream in=zip.getInputStream(entry)) { Files.copy(in,source,StandardCopyOption.REPLACE_EXISTING); }
      Info info;
      String before;
      MessageDigest d=MessageDigest.getInstance("SHA-256");
      try (TsFileSequenceReader r=new TsFileSequenceReader(source.toString());
           TsFileIOWriter writer=new TsFileIOWriter(output.toFile())) {
        info=inspect(r,p.target);
        writer.setGenerateTableSchema(false);
        writer.startChunkGroup(IDeviceID.Factory.DEFAULT_FACTORY.create(p.device));
        for (ChunkMetadata m:info.chunks) {
          Chunk chunk=r.readMemChunk(m);
          digest(d,m,chunk);
          writer.writeChunk(chunk,m);
        }
        writer.endChunkGroup();
        writer.endFile();
        before=hex(d.digest());
      }
      d=MessageDigest.getInstance("SHA-256");
      try (TsFileSequenceReader r=new TsFileSequenceReader(output.toString())) {
        Info after=inspect(r,p.target);
        if (!after.device.equals(p.device) || after.channels!=info.channels || after.points!=info.points || after.chunks.size()!=info.chunks.size())
          throw new IllegalStateException("Rebuilt schema does not match original");
        for (ChunkMetadata m:after.chunks) digest(d,m,r.readMemChunk(m));
      }
      if (!before.equals(hex(d.digest()))) throw new IllegalStateException("Numeric/time chunks changed");
      byte[] bytes=Files.readAllBytes(output);
      // Metadata is rebuilt, but also reject any residual literal old device ID.
      if (!info.device.equals(p.device) && new String(bytes,StandardCharsets.ISO_8859_1).contains(info.device))
        throw new IllegalStateException("Original device literal remains");
      return new Converted(p,bytes,p.target+"\t"+info.channels+"\t"+info.points+"\t"+info.chunks.size()+"\t"+info.first+"\t"+info.last+"\t"+before+"\t"+hex(MessageDigest.getInstance("SHA-256").digest(bytes)));
    } finally { Files.deleteIfExists(source); Files.deleteIfExists(output); }
  }
  public static void main(String[] args) throws Exception {
    if (args.length!=6) throw new IllegalArgumentException("input.zip private_plan.tsv output.partial.zip private_audit.tsv work_dir threads");
    java.nio.file.Path output=Paths.get(args[2]), work=Paths.get(args[4]);
    if (Files.exists(output)) throw new FileAlreadyExistsException(output.toString());
    Files.createDirectories(work);
    int threads=Integer.parseInt(args[5]);
    if (threads<1 || threads>8) throw new IllegalArgumentException("threads must be 1..8");
    ExecutorService pool=Executors.newFixedThreadPool(threads);
    boolean complete=false;
    try (ZipFile input=new ZipFile(args[0]);
         BufferedReader plan=Files.newBufferedReader(Paths.get(args[1]),StandardCharsets.UTF_8);
         ZipOutputStream out=new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(output)),StandardCharsets.UTF_8);
         BufferedWriter audit=Files.newBufferedWriter(Paths.get(args[3]),StandardCharsets.UTF_8)) {
      out.setLevel(1);
      audit.write("public_member\tchannels\tpoints_including_time\tchunks\tstart_tick\tend_tick\tpreserved_chunk_sha256\tpublic_tsfile_sha256\n");
      int done=0;
      while (true) {
        List<Future<Converted>> batch=new ArrayList<>();
        for (int i=0;i<threads*4;i++) {
          String line=plan.readLine();
          if (line==null) break;
          Plan p=new Plan(line);
          batch.add(pool.submit(()->convert(input,p,work)));
        }
        if (batch.isEmpty()) break;
        for (Future<Converted> future:batch) {
          Converted c=future.get();
          ZipEntry e=new ZipEntry(c.plan.target);
          e.setTime(315532800000L); // 1980-01-01 UTC; caller fixes JVM timezone
          e.setComment(null); e.setExtra(null);
          out.putNextEntry(e); out.write(c.bytes); out.closeEntry();
          audit.write(c.audit); audit.newLine();
          done++;
          if (done%1000==0) {audit.flush();System.out.println("VERIFIED "+done);System.out.flush();}
        }
      }
      audit.flush(); complete=true;
      System.out.println("COMPLETE "+done);
    } finally {
      pool.shutdownNow();
      if (!complete) System.err.println("INCOMPLETE: partial ZIP must not be distributed");
    }
  }
}
