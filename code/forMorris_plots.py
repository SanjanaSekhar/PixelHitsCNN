import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
from scipy import optimize

import ROOT
from ROOT import *

#gStyle.SetOptStat(1)
gROOT.SetBatch(1)
#gStyle.SetOptFit(1);


# double width count = 32967
# total count = 240077
n = [50966,43488,36097,28673] 
n_end = 80853

double_info = np.genfromtxt("txt_files/forMorris_doublepix.txt")
layer = double_info[:,0]
eta = double_info[:,1]
phi = double_info[:,2]

all_info = np.genfromtxt("txt_files/forMorris_allclusters.txt")
n_double = double_info[:,3]
layer_all = all_info[:,0]
eta_all = all_info[:,1]
phi_all = all_info[:,2]


print(eta.min(),eta.max())
print(phi.min(),phi.max())
print(eta[layer==1].shape,eta[layer==2].shape,eta[layer==3].shape,eta[layer==4].shape)
print(eta_all[layer_all==1].shape,eta_all[layer_all==2].shape,eta_all[layer_all==3].shape,eta_all[layer_all==4].shape)
for l in [1,2,3,4]:
	
	res = ROOT.TH1F("double pixels vs eta - BPIX L%i"%l,"double pixels vs eta - BPIX L%i"%l,50,eta.min(),eta.max())
	res.SetDirectory(0)
	res.Sumw2()
	for entry in eta[layer==l]:
		res.Fill(entry)

	res_all = ROOT.TH1F("all pixels vs eta - BPIX L%i"%l,"all pixels vs eta - BPIX L%i"%l,50,eta.min(),eta.max())
	res_all.SetDirectory(0)
	res_all.Sumw2()
	for entry in eta_all[layer_all==l]:
		res_all.Fill(entry)

	res_frac = ROOT.TH1F("double pixel fraction vs eta - BPIX L%i"%l,"double pixel fraction vs eta - BPIX L%i"%l,50,eta.min(),eta.max())
	res_frac.SetDirectory(0)
	res_frac.Sumw2()
	res_frac = res.Clone()
	res_frac.Divide(res_all)
	'''
	for i in range(1,51):
		a = res.GetBinContent(i)
		b = res_all.GetBinContent(i)
		if b!=0: res_frac.SetBinContent(i,a/b)
		else: res_frac.SetBinContent(i,0.)
		a_err = res.GetBinError(i)
		b_err = res_all.GetBinError(i)
		res_frac.SetBinError(i,0.)
	'''
	canvas = ROOT.TCanvas (" canvas ","canvas",1700,1000)
	canvas.cd()
	res_frac.SetMarkerColor(ROOT.kRed);
	res_frac.SetMarkerStyle(20)
	res_frac.SetMarkerSize(1)
	res_frac.SetLineColor(ROOT.kRed)
	res_frac.GetXaxis().SetTitle("eta")
	res_frac.GetYaxis().SetTitle("Fraction of double width pixels")
	res_frac.SetTitle("Fraction of double width pixels vs track eta - BPIX L%i"%l)
	res_frac.Draw("pe")
	canvas.Print("plots/forMorris/dpix_eta_%s.png"%l)

	res = ROOT.TH1F("double pixels vs phi - BPIX L%i"%l,"double pixels vs phi - BPIX L%i"%l,50,phi.min(),phi.max())
	res.SetDirectory(0)
	res.Sumw2()
	for entry in phi[layer==l]:
		res.Fill(entry)

	res_all = ROOT.TH1F("all pixels vs phi - BPIX L%i"%l,"all pixels vs phi - BPIX L%i"%l,50,phi.min(),phi.max())
	res_all.SetDirectory(0)
	res_all.Sumw2()
	for entry in phi_all[layer_all==l]:
		res_all.Fill(entry)

	res_frac = ROOT.TH1F("double pixel fraction vs phi - BPIX L%i"%l,"double pixel fraction vs phi - BPIX L%i"%l,50,phi.min(),phi.max())
	res_frac.SetDirectory(0)
	res_frac.Sumw2()
	res_frac = res.Clone()
	res_frac.Divide(res_all)
	'''
	for i in range(1,51):
		a = res.GetBinContent(i)
		b = res_all.GetBinContent(i)
		if b!=0: res_frac.SetBinContent(i,a/b)
		else: res_frac.SetBinContent(i,0.)
		a_err = res.GetBinError(i)
		b_err = res_all.GetBinError(i)
		res_frac.SetBinError(i,0.)
	'''
	canvas = ROOT.TCanvas (" canvas ","canvas",1700,1000)
	canvas.cd()
	res_frac.SetMarkerColor(ROOT.kRed);
	res_frac.SetMarkerStyle(20)
	res_frac.SetMarkerSize(1)
	res_frac.SetLineColor(ROOT.kRed)
	res_frac.GetXaxis().SetTitle("phi")
	res_frac.GetYaxis().SetTitle("Fraction of double width pixels")
	res_frac.SetTitle("Fraction of double width pixels vs track phi - BPIX L%i"%l)
	res_frac.Draw("pe")
	canvas.Print("plots/forMorris/dpix_phi_%s.png"%l)

res = ROOT.TH1F("double pixels vs eta - FPIX","double pixels vs eta - FPIX",50,eta.min(),eta.max())
res.SetDirectory(0)
res.Sumw2()
for entry in eta[layer==9]:
	res.Fill(entry)

res_all = ROOT.TH1F("all pixels vs eta - FPIX","all pixels vs eta - FPIX",50,eta.min(),eta.max())
res_all.SetDirectory(0)
res_all.Sumw2()
for entry in eta_all[layer_all==9]:
	res_all.Fill(entry)

res_frac = ROOT.TH1F("double pixel fraction vs eta - FPIX","double pixel fraction vs eta - FPIX",50,eta.min(),eta.max())
res_frac.SetDirectory(0)
res_frac.SetDirectory(0)
res_frac.Sumw2()
res_frac = res.Clone()
res_frac.Divide(res_all)
'''
for i in range(1,51):
	a = res.GetBinContent(i)
	b = res_all.GetBinContent(i)
	if b!=0: res_frac.SetBinContent(i,a/b)
	else: res_frac.SetBinContent(i,0.)
	a_err = res.GetBinError(i)
	b_err = res_all.GetBinError(i)
	res_frac.SetBinError(i,0.)
'''
canvas = ROOT.TCanvas (" canvas ","canvas",1700,1000)
canvas.cd()
res_frac.SetMarkerColor(ROOT.kRed);
res_frac.SetMarkerStyle(20)
res_frac.SetMarkerSize(1)
res_frac.SetLineColor(ROOT.kRed)
res_frac.GetXaxis().SetTitle("eta")
res_frac.GetYaxis().SetTitle("Fraction of double width pixels")
res_frac.SetTitle("Fraction of double width pixels vs track eta - FPIX")
res_frac.Draw("pe")
canvas.Print("plots/forMorris/dpix_eta_fpix.png")

res = ROOT.TH1F("double pixels vs phi - FPIX","double pixels vs phi - FPIX",50,phi.min(),phi.max())
res.SetDirectory(0)
res.Sumw2()
for entry in phi[layer==9]:
	res.Fill(entry)

res_all = ROOT.TH1F("all pixels vs phi - FPIX","all pixels vs phi - FPIX",50,phi.min(),phi.max())
res_all.SetDirectory(0)
res_all.Sumw2()
for entry in phi_all[layer_all==9]:
	res_all.Fill(entry)

res_frac = ROOT.TH1F("double pixel fraction vs phi - FPIX","double pixel fraction vs phi - FPIX",50,phi.min(),phi.max())
res_frac.SetDirectory(0)
res_frac.SetDirectory(0)
res_frac.Sumw2()
res_frac = res.Clone()
res_frac.Divide(res_all)
'''
for i in range(1,51):
	a = res.GetBinContent(i)
	b = res_all.GetBinContent(i)
	if b!=0: res_frac.SetBinContent(i,a/b)
	else: res_frac.SetBinContent(i,0.)
	a_err = res.GetBinError(i)
	b_err = res_all.GetBinError(i)
	res_frac.SetBinError(i,0.)
'''
canvas = ROOT.TCanvas (" canvas ","canvas",1700,1000)
canvas.cd()
res_frac.SetMarkerColor(ROOT.kRed);
res_frac.SetMarkerStyle(20)
res_frac.SetMarkerSize(1)
res_frac.SetLineColor(ROOT.kRed)
res_frac.GetXaxis().SetTitle("phi")
res_frac.GetYaxis().SetTitle("Fraction of double width pixels")
res_frac.SetTitle("Fraction of double width pixels vs track phi - FPIX")
res_frac.Draw("pe")
canvas.Print("plots/forMorris/dpix_phi_fpix.png")
